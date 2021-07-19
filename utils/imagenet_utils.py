def imagenet_eval(results, ground_truths, labels=None):
    top_1_sum, top_5_sum = 0, 0

    res_len = len(results) * len(results[0][0][0])

    if labels is not None:
        res_labels = list()

    for result, gt in zip(results, ground_truths):
        result = result[0][0]

        for tensor in result:
            tensor_sort = sorted(enumerate(tensor), key=lambda x:x[1])
            tensor_sort = list(zip(*tensor_sort))[0][-5:]

            if gt in tensor_sort:
                top_5_sum += 1
                
                if gt == tensor_sort[-1]:
                    top_1_sum += 1

            if labels is not None:
                res_labels.append((list(map(lambda x:(tensor[x], x, labels[x-1]), tensor_sort)), labels[gt-1]))


    if labels is None:
        return top_1_sum/res_len, top_5_sum/res_len
    else:
        return top_1_sum/res_len, top_5_sum/res_len, res_labels


def imagenet_label(results, labels):
    res_len = len(results) * len(results[0][0][0])
    
    res_labels = list()

    for result in results:
        result = result[0][0]

        for tensor in result:
            tensor = sorted(enumerate(filter(lambda x: x > 0.5, tensor)), key=lambda x:x[1])
            tensor = list(zip(*tensor))[0]

            res_labels.append(list(map(lambda x:labels[x-1], tensor)))

    return res_labels

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.read().split('\n')
    return labels
