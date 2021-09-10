# wav2vec2-malay

* Git Clone

  `git clone https://github.com/nam-nd-d3/wav2vec2-malay`
  
* install requirements

  `pip install -r requirements.txt`
  
* make sure you're logged into W&B

  `wandb login`
  
* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep

  `wandb agent --count count_number my_sweep_id`
