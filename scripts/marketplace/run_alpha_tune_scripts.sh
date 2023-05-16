CUDA_VISIBLE_DEVICES=6 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/fedex_for_cnn_cifar10.yaml &
CUDA_VISIBLE_DEVICES=6 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/fedex_for_cnn_cifar10_2_clients.yaml &
CUDA_VISIBLE_DEVICES=7 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10_3_clients_extrem.yaml &
CUDA_VISIBLE_DEVICES=7 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10.yaml &
CUDA_VISIBLE_DEVICES=7 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cifar10_2_clients.yaml &
CUDA_VISIBLE_DEVICES=6 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10_3_clients_zeros.yaml &
CUDA_VISIBLE_DEVICES=5 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10_2_clients_zeros.yaml &
CUDA_VISIBLE_DEVICES=5 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cifar10_2_clients_1000_0.yaml &
CUDA_VISIBLE_DEVICES=5 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cifar10_2_clients_1000_0_entropy_alpha.yaml &
CUDA_VISIBLE_DEVICES=6 python federatedscope/main.py --cfg scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10_3_clients_extrem_entropy_tau_alpha.yaml &


