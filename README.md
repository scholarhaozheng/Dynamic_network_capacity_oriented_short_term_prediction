This project is based on the implementations from [HCPLab-SYSU/HIAM](https://github.com/HCPLab-SYSU/HIAM) and [Yigang0622/Metro-Transfer-Algorithm](https://github.com/Yigang0622/Metro-Transfer-Algorithm).

## How to Run

### 1. Generate Domain Knowledge

Run the following command:

python generating_domain_knowledge_no_DO_clean.py

### 2. Train the Model

Execute the training script with the provided configuration file:

python train_button.py --config data/config/train_sh_dim76_units96_h4c512.yaml

### 3. Test the Model

1. Download the trained model and extract it to the directory: `data/checkpoint`.

2. Run the evaluation script:

    python evaluate_att.py --config data/checkpoint/eval_sh_dim76_units96_h4c512.yaml
