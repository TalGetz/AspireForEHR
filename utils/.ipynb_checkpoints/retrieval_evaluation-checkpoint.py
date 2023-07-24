import os
import pickle
import argparse


def load_data(folder_path):
    # iterate over all pickle files in the directory and load them
    # save the loaded data in a dictionary and return it
    results = {}
    for pickle_file in os.listdir(folder_path):
        if pickle_file.endswith(".pkl"):
            with open(os.path.join(folder_path,pickle_file), 'rb') as f:
                data = pickle.load(f)
                # the dictionary key is the name of the pickle file without the extension or the path
                results[pickle_file.split('.')[0]] = data
    return results

def get_labels(label_file_path):
    topic_article_to_label_dict = {}
    with open(label_file_path, 'r') as file:
        # read file in format "topic_id 0 article_id {0,1,2}"
        for line in file.readlines():
            topic_id, _, article_id, label = line.split()
            topic_article_to_label_dict[(topic_id, article_id)] = label
    return topic_article_to_label_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_folder', type=str, required=True, help='folder with pickle files to evaluate')
    parser.add_argument('--label_file', type=str, required=True, help='file which contains labels in TREC2016 format')
    parser.add_argument('--ascending', type=int, required=False, default=1, help='file which contains labels in TREC2016 format')
    parser.add_argument('-k', type=int, required=False, default=10, help='k for @k precision')
    args = parser.parse_args()
    retrieved_documents = load_data(args.pickle_folder)
    topic_article_to_label_dict = get_labels(args.label_file)
    precisions = []
    # iterate over all topics and calculate @10 precision
    for topic_id, retrieved_document_distance_tuples_for_topic in retrieved_documents.items():
        retrieved_document_distance_tuples_for_topic = sorted(retrieved_document_distance_tuples_for_topic, key=lambda x: x[1], reverse=args.ascending)
        retrieved_documents_for_topic = [document for document, _ in retrieved_document_distance_tuples_for_topic]
        # get the labels for the documents retrieved for this topic
        labels = [topic_article_to_label_dict.get((topic_id, article_id), 0) for article_id in retrieved_documents_for_topic]
        # calculate precision at 10
        labels = [int(int(label) > 0) for label in labels]
        at_percent = args.k
        precision_at_k = sum(labels[:at_percent]) / float(at_percent)
        precisions.append(precision_at_k)
        print('Topic {} has precision at {} of {}'.format(topic_id, at_percent, precision_at_k))
    print('Average precision at {} is {}'.format(at_percent, sum(precisions) / len(precisions)))

if __name__ == '__main__':
    main()
