import polars as pl
import pyarrow.parquet as pq

from pathlib import Path

__all__ = [
    'xmap_columns', 'cmap_columns',
    'load_data_from_dir', 'convert_to_single_table', 'get_distances', 'get_marker_positions'
]

xmap_columns = [
    "XmapEntryID",
    "QryContigID",
    "RefContigID",
    "QryStartPos",
    "QryEndPos",
    "RefStartPos",
    "RefEndPos",
    "Orientation",
    "Confidence",
    "HitEnum",
    "QryLen",
    "RefLen",
    "LabelChannel",
    "Alignment",
    "MapWt"
]

cmap_columns = [
    'CMapId',
    'ContigLength',
    'NumSites',
    'SiteID',
    'LabelChannel',
    'Position',
    'StdDev',
    'Coverage',
    'Occurrence',
    'GmeanSNR',
    'lnSNRsd'
]

def load_data_from_dir(base_dir):
    xmap = Path(base_dir) / 'exp_refineFinal1.xmap'
    qcmap = Path(base_dir) / 'exp_refineFinal1_q.cmap'
    stdout = Path(base_dir) / 'exp_refineFinal1.stdout'

    xmap = pl.read_csv(xmap, separator='\t', comment_char='#', has_header=False, new_columns=xmap_columns)
    qcmap = pl.read_csv(qcmap, separator='\t', comment_char='#', has_header=False, new_columns=cmap_columns)

    with open(stdout) as f:
        stdout = f.read()
    blobs = { 'exp_refineFinal1.stdout': stdout }
    
    return xmap, qcmap, blobs

def convert_to_single_table(xmap, qcmap, blobs):
    qcmap = qcmap.rename(
        {'CMapId': 'QryContigID'}
    ).group_by('QryContigID').agg(
        pl.col('ContigLength').first(), pl.col('Position')
    )

    xmap = xmap.select(
        pl.exclude('QryLen', 'XmapEntryID')
    ).with_columns(pl.col('Alignment')
        .str.strip_prefix('(')
        .str.strip_suffix(')')
        .str.split(')(')
        .list.eval(
            pl.element().str.splitn(',', 2).struct.rename_fields(['RefID', 'SiteID']).cast(pl.Int64)
        )
    )

    table = qcmap.join(xmap, on='QryContigID', how='outer').sort(by=['RefContigID', 'RefStartPos'])
    return table.to_arrow().replace_schema_metadata(blobs)

def get_distances(df, contig, start, end, startid, endid):
    return (
        df.filter(
            (pl.col('RefContigID') == contig) & (pl.col('RefStartPos') <= end) & (pl.col('RefEndPos') >= start)
        )
        .with_columns(
            pl.col('Alignment')
            .list.eval(
                pl.when(
                    pl.element().struct.field('RefID').is_in([startid, endid])
                ).then(
                    (pl.element().struct.field('SiteID') - 1)
                )
            )
            .list.drop_nulls()
            .alias('indices')
        )
        .select(
            pl.col('QryContigID'),
            pl.col('Position')
            .list.take(pl.col('indices'))
            .list.diff(null_behavior='drop')
            .alias('distance')
        )
        .explode('distance')
        .drop_nulls()
        .with_columns(
            pl.col('distance').abs().cast(pl.Int64)
        )
    )

def get_marker_positions(lf, contig, start, end, base_refid):

    def map_indices_to_positions(r):
        qry_contig_id, orientation, position, indices, base_idx, base_pos = r
        position = [(pos - base_pos) * orientation for pos in position]

        indices = indices if orientation == 1 else list(reversed(indices))

        tagged = []
        for idx, prev_idx in zip(indices, [-1] + indices):
            if (pos := position[idx]) >= 0:
                tagged.append({'position': pos, 'aligned': True})
            for i in range(prev_idx + 1, idx):
                if (pos := position[i]) > 0:
                    tagged.append({'position': pos, 'aligned': False})

        return (qry_contig_id, tagged)

    molecules = lf.filter(
        (pl.col('RefContigID') == contig) & (pl.col('RefStartPos') <= end) & (pl.col('RefEndPos') >= start)
    ).collect()

    positions = (
        molecules.select(
            pl.col('QryContigID'),
            pl.when(pl.col('Orientation') == '+').then(pl.lit(1)).otherwise(pl.lit(-1)).alias('Orientation'),
            pl.col('Position'),
            pl.col('Alignment').list.eval(pl.element().struct.field('SiteID') - 1).alias('indices'),
            pl.col('Alignment').list.eval(
                pl.element().filter(pl.element().struct.field('RefID') == base_refid).struct.field('SiteID') - 1
            ).alias('base_idx')
        )
        .explode('base_idx')
        .with_columns(
            pl.col('Position').list.get(pl.col('base_idx')).alias('base_pos')
        )
        .drop_nulls()
        .map_rows(map_indices_to_positions).rename({'column_0': 'QryContigID', 'column_1': 'Position'})
        .sort(
            pl.col('Position').list.eval(pl.element().struct.field('position')).list.max()
        )
    )

    return positions

def find_enclosing_site_ids(cmap_df, contig, start, end):
    contig = int(contig)
    start = int(start)
    end = int(end)

    start_id, start_pos = cmap_df.filter(
        (pl.col('CMapId') == contig) & (pl.col('Position') < end)
    ).select(
        pl.col('SiteID', 'Position')
    ).row(-1)
    
    end_id, end_pos = cmap_df.filter(
    (pl.col('CMapId') == contig) & (pl.col('Position') > start)
    ).select(
        pl.col('SiteID', 'Position')
    ).row(0)

    return start_id, end_id, start_pos, end_pos

def _convert(args):
    xmap, qcmap, blobs = load_data_from_dir(args.base_dir)
    table = convert_to_single_table(xmap, qcmap, blobs)

    pq.write_table(table, args.output, row_group_size=args.row_group_size)

def _distances(args):
    lf = pl.scan_parquet(args.input)
    distances = get_distances(lf, args.contig, args.start, args.end, args.startid, args.endid).collect()
    distances.write_csv(args.output, separator='\t')

def _marker_positions(args):
    lf = pl.scan_parquet(args.input)
    positions = get_marker_positions(lf, args.contig, args.start, args.end, args.base_refid)
    positions = positions.explode('Position').unnest('Position')
    positions.write_csv(args.output, separator='\t')

def _enclosing_site_ids(args):
    df = pl.read_parquet(args.input)
    r = find_enclosing_site_ids(df, args.contig, args.start, args.end)
    print('\t'.join(str(x) for x in r))
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('convert', help='Convert alignment files to a single parquet table.')
    subparser.add_argument('base_dir', help='Base directory containing exp_refineFinal1* files.')
    subparser.add_argument('output', help='Output Parquet file.')
    subparser.add_argument('-s', '--row-group-size', type=int, default=1000, help='Size of the row groups (default=1000)')
    subparser.set_defaults(func=_convert)

    subparser = subparsers.add_parser('distances', help='Generate distances table.')
    subparser.add_argument('input', help='Input parquet file.')
    subparser.add_argument('output', help='Output tsv file.')
    subparser.add_argument('contig', type=int)
    subparser.add_argument('start', type=int)
    subparser.add_argument('end', type=int)
    subparser.add_argument('startid', type=int)
    subparser.add_argument('endid', type=int)
    subparser.set_defaults(func=_distances)

    subparser = subparsers.add_parser('marker_positions', help='Generate aligned positions table.')
    subparser.add_argument('input', help='Input parquet file.')
    subparser.add_argument('output', help='Output tsv file.')
    subparser.add_argument('contig', type=int)
    subparser.add_argument('start', type=int)
    subparser.add_argument('end', type=int)
    subparser.add_argument('base_refid', type=int)
    subparser.set_defaults(func=_marker_positions)

    subparser = subparsers.add_parser('enclosing_site_ids', help='Find enclosing site IDs with positions.')
    subparser.add_argument('input', help='CMap file to query')
    subparser.add_argument('contig', type=int)
    subparser.add_argument('start', type=int)
    subparser.add_argument('end', type=int)
    subparser.set_defaults(func=_enclosing_site_ids)

    args = parser.parse_args()
    args.func(args)
