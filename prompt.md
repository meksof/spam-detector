#V2

1. We invode Typesafe client sequentially with local model when threshold is != than 2
2. We will ask a local model, already downloaded in `./model/` folder,  via ollama.
  - The model is already expert on spam detection.
  - The model will only add an output on stdout with an a verdict and a free-text explanation.
3. Improving local model performance: We are going to log metrics about threshold 2 cases. Making sure next time to improve model performance.

Elaborate a plan to implement the above requirements. The plan should include a breakdown of tasks, a description of the architecture, then saved into `implementation_plan.md`.