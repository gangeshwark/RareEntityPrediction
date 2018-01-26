# RareEntityPrediction

**You can Delete the following after you read it**

**Note**: create a `data` folder in your home(`~`) directory and place the `rare_entity` folder into it.

- **dataset folder**: used to process the raw data (`entities.txt` and `corpus.txt`, now I'm running the codes), which generate `all_data.json`, then load the json data, shuffle it and split it into three part *80%* for training, *10%* for dev and *10%* for test (codes are preparing).
```json
{
  "sentence": "<sentence contains **BLANK** to be filled>",
  "supplementary": "<supplementary sentence (maybe used)>",
  "candidates": "<list of candidates>",
  "descriptions": "<descriptions for each candidates>",
  "answer": "<Answer candidate>"
}
```

- **prepro folder**: used to read train/dev/test dataset and prepared vocabularies, embeddings, padded data and etc.