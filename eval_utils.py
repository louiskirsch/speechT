def extract_decoded_ids(sparse_tensor):
  ids = []
  last_batch_id = 0
  for i, index in enumerate(sparse_tensor.indices):
    batch_id, char_id = index
    if batch_id > last_batch_id:
      yield ids
      ids = []
      last_batch_id = batch_id
    ids.append(sparse_tensor.values[i])
  yield ids