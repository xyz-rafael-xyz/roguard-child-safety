"""Use the fixed three-expert recipe with the scope-distilled native checkpoint."""
from training import repaired_three_expert_v9 as experiment

def main():
    experiment.PARENTS = {102: 'scope-distillation-102'}
    experiment.STUDY = experiment.ROOT / 'distilled-three-expert-104'
    experiment.main()

if __name__ == '__main__':
    main()
