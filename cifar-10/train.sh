python -m src.main.py --data_root '.' --affix std
python -m src.main.py --data_root '.' -e 0.0157 -p 'linf' --adv_train --affix 'linf'
python -m src.main.py --data_root '.' -e 0.314 -p 'l2' --adv_train --affix 'l2'
