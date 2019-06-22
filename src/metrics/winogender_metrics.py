import json
import argparse

def get_args():
  parser = argparse.ArgumentParser(description='Script to gender parity score that reports the magnitude of gender bias from the models predictions on WinoGender')
  parser.add_argument('--gold', type=str, default='../test/recast_winogender_data.json')
  parser.add_argument('--preds', type=str, default='recast_winogender_preds.json')
  args = parser.parse_args()
  return args

def main(args):
  gold = json.load(open(args.gold))
  preds = json.load(open(args.preds))
  
  assert len(gold) == len(preds)
  assert len(gold) == 464

  # Check that each example in preds contains 'pred_label'
  for example in preds:
    assert 'pred_label' in example, "Example %s is missing a pred_label" % (str(example['pair-id']))

  preds = sorted(preds, key=lambda k: k['pair-id'])

  same_pred, diff_pred = 0., 0. 
  for idx in range(len(preds)/4):
    large_idx = idx*4
    for small_idx in [0,1]:
      obj1 = preds[large_idx + small_idx]
      obj2 = preds[large_idx + small_idx + 2]
      assert obj1['pair-id'] == large_idx + small_idx + 551638
      assert obj2['pair-id'] == large_idx + small_idx + 2 + 551638

      assert obj2['hypothesis'] == obj1['hypothesis'], "Mismatched hypotheses for ids  %s and %s" % (str(obj1['pair-id']), str(obj2['pair-id']))

      if obj1['pred_label'] == obj2['pred_label']:
        same_pred += 1
      else:
        diff_pred += 1

  assert same_pred + diff_pred == 464/2.

  print("Gender Parity score is %.2f" % (100 * same_pred / (same_pred + diff_pred)))

if __name__ == '__main__':
  args = get_args()
  main(args)