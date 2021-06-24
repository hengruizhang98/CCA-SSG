#  From Canonical Correlation Analysis to Self-supervised Graph Neural Networks
Code for CCA-SSG model proposed in the paper [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484).

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0


## Datasets

Citation Networks: 'Cora', 'Citeseer' and 'Pubmed'.

Co-occurence Networks: 'Amazon-Computer', 'Amazon-Photo', 'Coauthor-CS' and 'Coauthor-Physics'.

| Dataset          | # Nodes | # Edges | # Classes | # Features |
| ---------------- | ------- | ------- | --------- | ---------- |
| Cora             | 2,708   | 10,556  | 7         | 1,433      |
| Citeseer         | 3,327   | 9,228   | 6         | 3,703      |
| Pubmed           | 19,717  | 88,651  | 3         | 500        |
| Amazon-Computer  | 13,752  | 574,418 | 10        | 767        |
| Amazon-Photo     | 7,650   | 287,326 | 8         | 745        |
| Coauthor-CS      | 18,333  | 327,576 | 15        | 6,805      |
| Coauthor-Physics | 34,493  | 991,848 | 5         | 8,451      |

## Usage
To run the codes, use the following commands:
```python
# Cora
python main.py --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4

# Citeseer
python main.py --dataname citeseer --epochs 20 --n_layers 1 --lambd 5e-4 --dfr 0.0 --der 0.4 --lr2 1e-2 --wd2 1e-2

# Pubmed
python main.py --dataname pubmed --epochs 100 --lambd 1e-3 --dfr 0.3 --der 0.5 --lr2 1e-2 --wd2 1e-4

# Amazon-Computer
python main.py --dataname comp --epochs 50 --lambd 5e-4 --dfr 0.1 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Amazon-Photo
python main.py --dataname photo --epochs 50 --lambd 1e-3 --dfr 0.2 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Coauthor-CS
python main.py --dataname cs --epochs 50 --lambd 1e-3 --dfr 0.2 --lr2 5e-3 --wd2 1e-4 --use_mlp

# Coauthor-Physics
python main.py --dataname physics --epochs 100 --lambd 1e-3 --dfr 0.5 --der 0.5 --lr2 5e-3 --wd2 1e-4
```

## Reference
If our paper and code are useful for your research, please cite the following article:
```
@article{cca-ssg,
      title   = {From Canonical Correlation Analysis to Self-supervised Graph Neural Networks}, 
      author  = {Hengrui Zhang and Qitian Wu and Junchi Yan and David Wipf and Philip S. Yu},
      journal = {arXiv preprint arXiv:2106.12484},
      year    = {2021}
}
```