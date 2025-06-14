import argparse
from model_loading import load_model
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None, help="Use with desired config name to train the corresponding model")
    parser.add_argument("--eval", type=str, default=None, help="Use with desired config name to evaluate the corresponding model")
    parser.add_argument("--search_eval", type=str, default=None, help="Use with desired config name to evaluate the corresponding model (and search for best aggregation)")
    parser.add_argument("--visualize", type=str, help="Use with desired config name to visualize the training curve")
    parser.add_argument("--dataset", type=str, default="deepfakes", help="Dataset to use (deepfakes, f2f, or combined)")
    parser.add_argument("--checkpoint", type=str, default=None, help="'best' or 'epoch_X' or leave empty to train from scratch")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to sample from each video")
    parser.add_argument("--agg", type=str, default="50_vote", help="Aggregation method for evaluation (average or X_vote)")
    args = parser.parse_args()

    assert args.train or args.eval or args.search_eval, "--train, --eval, or --search_eval must be specified"

    if args.train:
        model = load_model(config_name=args.train, checkpoint=args.checkpoint)
        trainer = Trainer(model=model, dataset=args.dataset, num_frames=args.num_frames, epochs=args.num_epochs)
        trainer.train()
    elif args.eval:
        if args.checkpoint is None:
            args.checkpoint = "best"
        model = load_model(config_name=args.eval, checkpoint=args.checkpoint)
        evaluator = Evaluator(model=model, dataset=args.dataset, num_frames=args.num_frames)
        evaluator.evaluate(aggregation=args.agg)
    elif args.search_eval:
        if args.checkpoint is None:
            args.checkpoint = "best"
        model = load_model(config_name=args.search_eval, checkpoint=args.checkpoint)
        evaluator = Evaluator(model=model, dataset=args.dataset, num_frames=args.num_frames)
        evaluator.search_evaluate()
    elif args.visualize:
        visualizer = Visualizer(model_name=args.visualize)
        visualizer.plot(save=True, show=False)

if __name__ == "__main__":
    main()
