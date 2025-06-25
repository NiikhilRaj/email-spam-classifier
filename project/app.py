import argparse
import sys
import os

TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_model.py')
PREDICT_SCRIPT = os.path.join(os.path.dirname(__file__), 'predict.py')


def main():
    parser = argparse.ArgumentParser(description="Email Spam Classifier CLI")
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Predict command
    predict_parser = subparsers.add_parser(
        'predict', help='Predict spam/ham for input text')
    predict_parser.add_argument('text', nargs='+', help='Text(s) to classify')

    args = parser.parse_args()

    if args.command == 'train':
        print('Training model...')
        os.system(f'{sys.executable} {TRAIN_SCRIPT}')
    elif args.command == 'predict':
        from predict import predict
        results = predict(args.text)
        for text, label in zip(args.text, results):
            print(f"Text: {text}\nPrediction: {label}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
