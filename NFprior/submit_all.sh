# python train_NF_prior.py --submit --population-type uniform --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --population-type uniform --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type uniform --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --population-type gaussian --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --population-type gaussian --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type gaussian --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --population-type double_gaussian --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --population-type double_gaussian --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type double_gaussian --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --population-type uniform --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --population-type uniform --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type uniform --source-type nsbh --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --population-type gaussian --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --population-type gaussian --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type gaussian --source-type nsbh --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --population-type double_gaussian --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --population-type double_gaussian --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --population-type double_gaussian --source-type nsbh --eos-samples-name radio_chiEFT_NICER

#!/
### GW170817 ###
python train_NF_prior.py --submit --source-type GW170817 --eos-samples-name radio
python train_NF_prior.py --submit --source-type GW170817 --eos-samples-name radio_chiEFT
python train_NF_prior.py --submit --source-type GW170817 --eos-samples-name radio_chiEFT_NICER

### GW190425 ###
python train_NF_prior.py --submit --source-type GW190425 --eos-samples-name radio
python train_NF_prior.py --submit --source-type GW190425 --eos-samples-name radio_chiEFT
python train_NF_prior.py --submit --source-type GW190425 --eos-samples-name radio_chiEFT_NICER

### GW230529 ###
python train_NF_prior.py --submit --source-type GW230529 --eos-samples-name radio
python train_NF_prior.py --submit --source-type GW230529 --eos-samples-name radio_chiEFT
python train_NF_prior.py --submit --source-type GW230529 --eos-samples-name radio_chiEFT_NICER