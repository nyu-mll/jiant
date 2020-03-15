from sklearn import metrics
import numpy
import codecs
import os


def get_accuracy(p, len_sent):
    return float(len([a for a in p[:len_sent] if a])) / float(len_sent)


def get_eval(predictions, gs):
    y, y_ = [], []
    for p in predictions:
        y.extend(map(lambda x: list(x).index(x.max()), p))
    for g in gs:
        y_.extend(map(lambda x: 0 if list(x) == [1, 0] else 1, g))

    print(metrics.classification_report(y_, y))
    cm = metrics.confusion_matrix(y_, y)
    print(cm)

    p, r, f1, s = metrics.precision_recall_fscore_support(y_, y)
    report = "%s\n%s\n%s\n%s\n\n" % (str(p), str(r), str(f1), str(s))

    f1_pos = f1[0]

    return numpy.average(f1, weights=s), report, cm, f1_pos


def write_report(folder, report, cm, name):
    print("Storing reports...")
    with codecs.open(
        os.path.join(folder, "%s_report.txt" % name), "wb", "utf8"
    ) as store_rep_dev:
        store_rep_dev.write(report)
        store_rep_dev.write(str(cm) + "\n")
    print("Reports stored...")


def store_prediction(folder, lex, dic_inv, pred_dev, gold_dev, name):
    print("Storing labelling results for dev set...")
    with codecs.open(
        os.path.join(folder, "best_%s.txt" % name), "wb", "utf8"
    ) as store_pred:
        for s, y_sys, y_hat in zip(lex, pred_dev, gold_dev):
            s = [dic_inv["idxs2w"][w] if w in dic_inv["idxs2w"] else "<UNK>" for w in s]
            assert len(s) == len(y_sys) == len(y_hat)
            for _word, _sys, gold in zip(s, y_sys, y_hat):
                _p = list(_sys).index(_sys.max())
                _g = 0 if list(gold) == [1, 0] else 1
                store_pred.write("%s\t%s\t%s\n" % (_word, _g, _p))
            store_pred.write("\n")
