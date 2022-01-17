import math
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from ast import literal_eval
from tqdm import tqdm

from topic_extraction.data.similarity import word_similarity_score, cosine_similarity_score
from topic_extraction.inference.bert_embedder import BertEmbedder


METRICS = ["Precision","MRR","MAP","nDCG"]



def parse_cmd():
    parser = ArgumentParser(description="Evaluate some Keyword Extraction method")
    parser.add_argument("--lang", type=str, help="language (should be DA, SV, NO or EN)")
    parser.add_argument("--sources", nargs='+', required=True,
                        help="paths to source files containing the predictions to evaluate (should be *.csv)")
    parser.add_argument("--output_files", nargs='+', help="path to output files (should be *.csv)")
    parser.add_argument("--pred_col", type=str, help="which column contains the predictions to evaluate")
    parser.add_argument("--keywords_col", type=str, default="keywords",
                        help="which column contains the human-annotated keywords")
    parser.add_argument('--all_preds', action='store_true',
                        help='evaluate all available predictions from the CSV')
    parser.add_argument("--metrics", nargs='+', default="all",
                        help="which metrics to use for evaluation (Precision, F1, 1st_correct_pred, avg_precision, nDCG)")
    parser.add_argument("--num_preds", type=int, default=10,
                        help="number of predictions to evaluate per document")
    return parser.parse_args()


# Computes mean similarity score between 1 prediction and Y annotations
def relevance_score(pred, annotations, embedder) -> float:
    scores = []
    for ann in annotations:
        scores.append(cosine_similarity_score(ann, pred, embedder))
    return np.mean(scores)


def main():
    args = parse_cmd()

    for src_idx, source in enumerate(args.sources):
        print(f'--> Computing metrics for {source}')
        df = pd.read_csv(source, index_col=0, encoding="utf-8")
        results = {}

        # which metrics to use for evaluation
        if args.metrics=='all':
            metrics = METRICS
        else:
            metrics = args.metrics
    
        # select which predictions to evaluate
        if args.all_preds:
            ke_methods = [i.replace('_keywords', '') for i in list(df.columns.values) if i.endswith('_keywords') and 'num_' not in i]
        else:
            ke_methods = [args.pred_col.replace('_keywords', '')]
        
        for metric in metrics:

            # Precision
            # num. annotations is variable for each dataset, could be a BIAS
            # we count the number of matches per annotation and compute the mean
            # we sum all means-per-doc and divide them by num_preds*num_docs
            # result: sum(mean_matches_per_doc) / (num_preds*num_docs)
            if metric.lower()=='precision':
                results['TP/Results'] = []
                results['wTP/Results'] = []
                results['Precision'] = []
                results['wPrecision'] = []
                for ke_method in ke_methods:
                    total_matches = 0
                    total_w_matches = 0.0
                    for idx, row in df.iterrows():
                        X = [i.lower() for i in literal_eval(row[ke_method+'_keywords'])]
                        Y = [i.lower() for i in literal_eval(row[args.keywords_col])]
                        cum = []
                        w_cum = []
                        for y in Y:
                            matches = 0
                            w_matches = 0.0
                            for x in X[:args.num_preds]:
                                if x==y:
                                    matches += 1
                                w_matches += word_similarity_score(x, y)
                            cum.append(matches)
                            w_cum.append(w_matches)
                        total_matches += int(np.mean(cum))
                        total_w_matches += np.mean(w_cum)
                    precision = total_matches/(args.num_preds*df.shape[0])
                    w_precision = total_w_matches/(args.num_preds*df.shape[0])
                    results['TP/Results'].append((total_matches, args.num_preds*df.shape[0]))
                    results['wTP/Results'].append((int(total_w_matches), args.num_preds*df.shape[0]))
                    results['Precision'].append(float(np.round(precision, decimals=3)))
                    results['wPrecision'].append(float(np.round(w_precision, decimals=3)))
        
            # MRR
            # num. annotations is variable for each dataset, could be a BIAS
            # we average the rank of the 1st match per annotation -> avg(1/1st_match)
            # result: sum(1/avg_1st_match_position_per_doc) / num_docs
            elif metric.lower()=='mrr':
                results['MRR'] = []
                results['MRR_var'] = []
                results['wMRR'] = []
                results['wMRR_var'] = []
                for ke_method in ke_methods:
                    acc = []
                    w_acc = []
                    for idx, row in df.iterrows():
                        X = [i.lower() for i in literal_eval(row[ke_method+'_keywords'])]
                        Y = [i.lower() for i in literal_eval(row[args.keywords_col])]
                        cum = []
                        w_cum = []
                        for y in Y:
                            # exact matches
                            pos = 0
                            for i, x in enumerate(X[:args.num_preds]):
                                if x==y:
                                    pos = i+1
                                    break
                            cum.append(pos)
                            # partial word matches
                            w_pos = 0
                            for i, x in enumerate(X[:args.num_preds]):
                                if word_similarity_score(x, y)>0.0:
                                    w_pos = i+1
                                    break
                            w_cum.append(w_pos)
                        cum = [1/item if item>0 else 0 for item in cum]
                        w_cum = [1/item if item>0 else 0 for item in w_cum]
                        acc.append(np.mean(cum))
                        w_acc.append(np.mean(w_cum))
                    mrr = np.mean(acc)
                    mrr_var = np.var(acc)
                    w_mrr = np.mean(w_acc)
                    w_mrr_var = np.var(w_acc)
                    results['MRR'].append(float(np.round(mrr, decimals=3)))
                    results['MRR_var'].append(float(np.round(mrr_var, decimals=3)))
                    results['wMRR'].append(float(np.round(w_mrr, decimals=3)))
                    results['wMRR_var'].append(float(np.round(w_mrr_var, decimals=3)))

            # MAP
            # num. annotations is variable for each dataset, could be a BIAS
            # we average the average sub-list precision score per annotation
            # result: sum(mean_per_doc(sub-list_accuracy_per annotation/matches_per_annotation)) / num_docs
            elif metric.lower()=='map':
                results['MAP'] = []
                results['MAP_var'] = []
                results['wMAP'] = []
                results['wMAP_var'] = []
                for ke_method in ke_methods:
                    avgs = []
                    w_avgs = []
                    for idx, row in df.iterrows():
                        X = [i.lower() for i in literal_eval(row[ke_method+'_keywords'])]
                        Y = [i.lower() for i in literal_eval(row[args.keywords_col])]
                        cum = []
                        w_cum = []
                        for y in Y:
                            acc = 0.0
                            w_acc = 0.0
                            matches = 0
                            w_matches = 0
                            pos = 1
                            for x in X[:args.num_preds]:
                                if x==y:
                                    matches += 1
                                    acc += matches / pos
                                w_sim = word_similarity_score(x, y)
                                if w_sim>0.0:
                                    w_matches += 1
                                    w_acc += w_matches / pos
                                pos += 1
                            cum.append(acc/matches if matches > 0 else 0.0)
                            w_cum.append(w_acc/w_matches if w_matches > 0 else 0.0)
                        avgs.append(np.mean(cum))
                        w_avgs.append(np.mean(w_cum))
                    map = np.mean(avgs)
                    map_var = np.var(avgs)
                    w_map = np.mean(w_avgs)
                    w_map_var = np.var(w_avgs)
                    results['MAP'].append(float(np.round(map, decimals=3)))
                    results['MAP_var'].append(float(np.round(map_var, decimals=3)))
                    results['wMAP'].append(float(np.round(w_map, decimals=3)))
                    results['wMAP_var'].append(float(np.round(w_map_var, decimals=3)))
            
            # PbS & nDCG
            # num. annotations is variable for each dataset, could be a BIAS
            # we average cosine similarity per annotation
            # PbS <=> nDCG not considering ranking
            elif metric.lower()=='ndcg':
                embedder = BertEmbedder(args.lang.lower())
                results['PbS'] = []
                results['PbS_var'] = []
                results['nDCG'] = []
                results['nDCG_var'] = []
                for ke_method in ke_methods:
                    pbss = []
                    ndcgs = []
                    print(f'Computing PbS & nDCG for {ke_method} method...')
                    for idx, row in tqdm(df.iterrows()):
                        X = [i.lower() for i in literal_eval(row[ke_method+'_keywords'])]
                        Y = [i.lower() for i in literal_eval(row[args.keywords_col])]
                        #if len(Y)>args.num_preds: Y = Y[:args.num_preds]
                        ipbs = args.num_preds
                        idcg = sum([(1/(math.log(i+1,2))) for i,_ in enumerate(range(args.num_preds),start=1)])
                        relevances = [relevance_score(pred,Y,embedder) for pred in X[:args.num_preds]]
                        pbs = sum(relevances)
                        dcg = sum([rel/math.log(i+1,2) if rel>0.0 else 0.0 for i,rel in enumerate(relevances,start=1)])
                        pbss.append(pbs/ipbs)
                        ndcgs.append(dcg/idcg)
                    avg_pbs = np.mean(pbss)
                    var_pbs = np.var(pbss)
                    ndcg = np.mean(ndcgs)
                    ndcg_var = np.var(ndcgs)
                    results['PbS'].append(float(np.round(avg_pbs, decimals=3)))
                    results['PbS_var'].append(float(np.round(var_pbs, decimals=3)))
                    results['nDCG'].append(float(np.round(ndcg, decimals=3)))
                    results['nDCG_var'].append(float(np.round(ndcg_var, decimals=3)))
        
        data = [results[key] for key in results.keys()]
        df = pd.DataFrame(data,columns=ke_methods)
        df['metric'] = results.keys()
        df.set_index('metric', inplace=True)

        if args.output_files is None:
            df.to_csv(source.replace('.csv', '_report.csv'))
        else:
            df.to_csv(args.output_files[src_idx])



if __name__ == '__main__':
    main()
