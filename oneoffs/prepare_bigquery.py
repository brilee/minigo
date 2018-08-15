'''
Script to process debug SGFs for upload to BigQuery.

Handles one directory per invocation, for easy sharding of work.

Usage (single file):
BOARD_SIZE=19 python oneoffs/prepare_bigquery.py \
    --sgf_dir=gs://$BUCKET_NAME/sgf/full/2018-07-20-00/
    --output_dir=gs://$BUCKET_NAME/bigquery/

Usage (multiple files)
export BOARD_SIZE=19
export BUCKET_NAME=...
gsutil ls gs://$BUCKET_NAME/sgf/full | \
    xargs -I '{}' -n 1 -P 32 \
    python3 oneoffs/prepare_bigquery.py \
    --sgf_dir='{}' \
    --output_dir=gs://tensor-go-minigo-v9-19/bigquery

The load commands look like:

export PROJECT_ID=blah
export DATASET_ID=minigo_v9_19

bq mk --project_id=$PROJECT_ID $DATASET_ID

bq load --project_id=$PROJECT_ID \
    --replace \
    --source_format=NEWLINE_DELIMITED_JSON \
    $PROJECT_ID:$DATASET_ID.games \
    gs://$BUCKET_NAME/bigquery/games/* \
    oneoffs/bigquery_games_schema.json

bq load --project_id=$PROJECT_ID \
    --replace \
    --source_format=NEWLINE_DELIMITED_JSON \
    $PROJECT_ID:$DATASET_ID.moves \
    gs://$BUCKET_NAME/bigquery/moves/* \
    oneoffs/bigquery_moves_schema.json

'''
import sys
sys.path.insert(0, '.')

import collections
import json
import os
import random
import re

from absl import app, flags
from tensorflow import gfile
from tqdm import tqdm

import coords
import go
import sgf_wrapper
import utils

DebugRow = collections.namedtuple('DebugRow', [
    'move', 'action', 'Q', 'U', 'prior', 'orig_prior', 'N', 'soft_N', 'p_delta', 'p_rel'])

flags.DEFINE_float("sampling_frac", 0.05, "Fraction of SGFs to process.")

flags.DEFINE_string("sgf_dir", None, "base directory for full debug minigo sgfs")

flags.DEFINE_string("output_dir", None, "output directory for bigquery data")

FLAGS = flags.FLAGS

FILENAME_RE = re.compile(r'(\d+)-tpu-player-deployment-(.*)\.sgf')

def _get_timestamp_worker_id(filename):
    match = FILENAME_RE.search(filename)
    if match and len(match.groups()) == 2:
        timestamp, worker_id = match.groups()
        return int(timestamp), worker_id
    raise ValueError("Couldn't parse {}".format(filename))

def _get_model_num(model_comment):
    '''
    Takes a comment string like
    "models:gs://some-bucket/work_dir/model.ckpt-139529,gs://some-bucket/work_dir/model.ckpt-123"
    and turns it into
    139529
    '''
    try:
        matches = re.findall('(\d+)($|,)', model_comment)
        # => [('139529', ','), ('123', '')]
        models = [int(match[0]) for match in matches]
        # => [139529, 123]
        return max(models)
    except:
        print(model_comment)
        return 0


def list_sgfs(path):
    sgfs = gfile.Glob(os.path.join(path, '*.sgf'))
    return [sgf for sgf in sgfs if random.random() < FLAGS.sampling_frac]

def process_directory(sgf_dir_path, output_dir_path):
    sgf_dirname = os.path.basename(sgf_dir_path)

    games_path = os.path.join(output_dir_path, 'games', sgf_dirname)
    moves_path = os.path.join(output_dir_path, 'moves', sgf_dirname)

    with gfile.GFile(games_path, 'w') as game_f, \
            gfile.GFile(moves_path, 'w') as move_f:
        for sgf_path in tqdm(list_sgfs(sgf_dir_path)):
            game_data, move_data = extract_data(sgf_path)
            game_f.write(json.dumps(game_data) + '\n')
            move_f.write(''.join(json.dumps(move_datum) + '\n'
                                 for move_datum in move_data))


def extract_data(filename):
    with gfile.GFile(filename) as f:
        contents = f.read()
    root_node = sgf_wrapper.get_sgf_root_node(contents)
    game_data = extract_game_data(filename, root_node)
    move_data = extract_move_data(
        root_node,
        game_data['worker_id'],
        game_data['completed_time'])
    return game_data, move_data


def extract_game_data(gcs_path, root_node):
    props = root_node.properties
    result = sgf_wrapper.sgf_prop(props.get('RE', ''))
    board_size = int(sgf_wrapper.sgf_prop(props.get('SZ')))
    if board_size != go.N:
        raise Exception("Game file uses BOARD_SIZE={} but script was run with BOARD_SIZE={}".format(board_size, go.N))
    value = utils.parse_game_result(result)
    was_resign = '+R' in result

    completed_time, worker_id = _get_timestamp_worker_id(gcs_path)
    sgf_url = gcs_path
    first_comment_node_lines = root_node.next.properties['C'][0].split('\n')
    # in-place edit to comment node so that first move's comment looks
    # the same as all the other moves.
    root_node.next.properties['C'][0] = '\n'.join(first_comment_node_lines[1:])
    resign_threshold = float(first_comment_node_lines[0].split()[-1])


    return {
        'worker_id': worker_id,
        'completed_time': completed_time,
        'board_size': board_size,
        'result_str': result,
        'value': value,
        'was_resign': was_resign,
        'sgf_url': sgf_url,
        'resign_threshold': resign_threshold,
    }


def extract_move_data(root_node, worker_id, completed_time):
    current_node = root_node.next
    move_data = []
    move_num = 1
    while current_node is not None:
        props = current_node.properties
        if 'B' in props:
            to_play = 1
            move_played = props['B'][0]
        elif 'W' in props:
            to_play = -1
            move_played = props['W'][0]
        else:
            import pdb; pdb.set_trace()
        move_played = coords.to_flat(coords.from_sgf(move_played))
        post_Q, model_num, debug_rows = parse_comment_node(props['C'][0])

        def get_row_data(debug_row):
            column_names = ["prior", "orig_prior", "N", "soft_N"]
            return [getattr(debug_row, field) for field in column_names]

        if not debug_rows:
            row_data = [None, None, None, None]
        else:
            row_data = get_row_data(debug_rows[0])
        policy_prior, policy_prior_orig, mcts_visits, mcts_visits_norm = row_data


        move_data.append({
            'worker_id': worker_id,
            'completed_time': completed_time,
            'model_num': model_num,
            'move_num': move_num,
            'turn_to_play': to_play,
            'move': move_played,
            'move_kgs': coords.to_kgs(coords.from_flat(move_played)),
            'post_Q': post_Q,
            'policy_prior': policy_prior,
            'policy_prior_orig': policy_prior_orig,
            'mcts_visit_counts': mcts_visits,
            'mcts_visit_counts_norm': mcts_visits_norm,
        })
        move_num += 1
        current_node = current_node.next
    return move_data


def parse_comment_node(comment):
    # Example of a comment node. The resign threshold line appears only
    # for the first move in the game; it gets preprocessed by extract_game_data
    """
    Resign Threshold: -0.88
    models:gs://some-bucket/work_dir/model.ckpt-139529
    -0.0662
    D4 (100) ==> D16 (14) ==> Q16 (3) ==> Q4 (1) ==> Q: -0.07149
    move: action Q U P P-Dir N soft-N p-delta p-rel
    D4 : -0.028, -0.048, 0.020, 0.048, 0.064, 100 0.1096 0.06127 1.27
    D16 : -0.024, -0.043, 0.019, 0.044, 0.059, 96 0.1053 0.06135 1.40
    """

    lines = comment.split('\n')
    if lines[0].startswith('Resign'):
        lines = lines[1:]

    model_num = _get_model_num(lines[0])
    post_Q = float(lines[1])
    debug_rows = []
    comment_splitter = re.compile(r'[ :,]')
    for line in lines[4:]:
        if not line:
            continue
        columns = comment_splitter.split(line)
        columns = list(filter(bool, columns))
        coord, *other_columns = columns
        coord = coords.to_flat(coords.from_kgs(coord))
        debug_rows.append(DebugRow(coord, *map(float, other_columns)))
        if True:
            break
    return post_Q, model_num, debug_rows


def main(argv):
    del argv
    sgf_dir = FLAGS.sgf_dir
    if sgf_dir.endswith('/'):
        sgf_dir = sgf_dir.rstrip('/')
    process_directory(sgf_dir, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
