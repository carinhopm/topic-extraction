import pandas as pd
from argparse import ArgumentParser

from topic_extraction.inference import KEPredictor



def parse_cmd():
    parser = ArgumentParser(description="Predict with Topic Extraction model and save the results in one of the supported formats.")
    parser.add_argument("--lang", type=str, help="language (should be DA, SV, NO or EN)")
    parser.add_argument("--sources", nargs='+', required=True,
                        help="path to source files containing texts (should be *.csv)")
    parser.add_argument("--output_files", nargs='+', help="path to output files (should be *.csv)")
    parser.add_argument("--ke_method", type=str, help="KE method to use for prediction", default="novel")
    parser.add_argument("--keyword_len", type=int, default=4, help="max. number of words per keyphrase")
    parser.add_argument("--num_keywords", type=int, default=10, help="number of keywords per article")
    parser.add_argument("--input_col", type=str, default="text", help="which column should be considered as input")
    parser.add_argument('--keep_text', help='Keep article body into results', action='store_true')
    parser.add_argument('--ke_score', help='Return KE score per keyword', action='store_true')
    parser.add_argument('--f_sem', help='Semantic factor value', type=float, default=0.9)

    return parser.parse_args()


def main():
    args = parse_cmd()

    ke_model = KEPredictor([args.lang.lower()], f_sem=args.f_sem)

    for src_idx, source in enumerate(args.sources):
        print(f'--> Computing predictions for {source}')
        df = pd.read_csv(source, index_col=0, encoding="utf-8")

        keywords = ke_model.predict(df[args.input_col].tolist(), 
                                    args.lang.lower(),
                                    keyword_len=args.keyword_len,
                                    num_keywords=args.num_keywords, 
                                    incl_score=args.ke_score)
    
        df[args.ke_method+'_keywords'] = keywords
        df['num_'+args.ke_method+'_keywords'] = df.apply(lambda row: len(row[args.ke_method+'_keywords']), axis=1)
        if not args.keep_text:
            del df['text']
        
        if args.output_files is None:
            df.to_csv(source.replace('.csv', '_predictions.csv'))
        else:
            df.to_csv(args.output_files[src_idx])



if __name__ == '__main__':
    main()
