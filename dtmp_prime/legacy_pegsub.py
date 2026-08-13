"""
Funções legadas herdadas do aplicativo web original (pegsub).
NÃO fazem parte do pipeline do DTMP-Prime e não são chamadas por
nenhum módulo deste projeto.

Foram isoladas aqui porque liam e escreviam em diretórios que não existem na
estrutura atual (``Sequence/``, ``Position/``, ``pegsub/static/``) com caminhos
fixos no código.

Os caminhos agora são parâmetros explícitos, sem default oculto. Preservadas
para rastreabilidade da reconstrução; remova-as quando não forem mais úteis
como referência.
"""
import time

import torch
import pandas as pd

def transformer_predictor_order3_file(transformer, X_test, batch_size_test, device,
                                      pegrna_table, result_out):
    # test
    transformer.eval()
    n = len(X_test)
    batch_num = n // batch_size_test + 1
    print("batch_num", batch_num)
    start = time.time()
    with torch.no_grad():
        outputs = []
        att_weights = []
        best = {'max': 0, 'Target': 'ACGT', 'PBS': 'ACGT', 'RT': 'ACGT'}
        for i in range(batch_num):
            start_i = i * batch_size_test
            end_i = start_i + batch_size_test
            xb = X_test.iloc[start_i:end_i, :]

            input = (torch.tensor(list(xb["Target"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o3"]), device=device, dtype=torch.long))
            # torch.tensor(list(xb["Other"]), device=device, dtype=torch.float32))
            output_b, att_weights_b = transformer(input)
            output_b = output_b.squeeze(-1).cpu().numpy().tolist()
            print("att_weights", att_weights_b)
            # group max
            index_max = output_b.index(max(output_b))  # index max
            best['max'] = max(output_b)

            df = pd.read_table(pegrna_table, header=0)  # re-load

            dfone = df.iloc[index_max]
            best['Target'] = dfone['Target(47bp)']
            best['PBS'] = dfone['PBS']
            best['RT'] = dfone['RT']

            outputs = outputs + output_b
            # print(outputs)
            # df["Efficiency"] = outputs
            # df.to_csv(f'Sequence/result.txt', sep="\t")
            if not att_weights:
                att_weights = att_weights_b
            else:
                for j in range(len(att_weights_b)):
                    att_weights[j] = torch.cat((att_weights[j], att_weights_b[j]), 0)
        print(outputs)
        df["EditingScore"] = outputs
        df.to_csv(result_out, sep="\t")
    print(f'Predicting time: {time.time() - start}')
    return best


# update
def transformer_predictor_order3_file_update(transformer, X_test, batch_size_test, device, TOP_N, df):
    # test
    transformer.eval()
    n = len(X_test)
    batch_num = n // batch_size_test + 1
    print("batch_num", batch_num)
    start = time.time()
    with torch.no_grad():
        outputs = []
        att_weights = []
        best = {'max': 0, 'Target': 'ACGT', 'PBS': 'ACGT', 'RT': 'ACGT'}
        for i in range(batch_num):
            start_i = i * batch_size_test
            end_i = start_i + batch_size_test
            xb = X_test.iloc[start_i:end_i, :]

            input = (torch.tensor(list(xb["Target"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o3"]), device=device, dtype=torch.long))
            # torch.tensor(list(xb["Other"]), device=device, dtype=torch.float32))
            output_b, att_weights_b = transformer(input)
            output_b = output_b.squeeze(-1).cpu().numpy().tolist()
            # print("att_weights", att_weights_b)

            outputs = outputs + output_b
            if not att_weights:
                att_weights = att_weights_b
            else:
                for j in range(len(att_weights_b)):
                    att_weights[j] = torch.cat((att_weights[j], att_weights_b[j]), 0)
        # df = pd.read_table(f'Sequence/pegRNA_update.User.txt', header=0)
        # df = pd.read_table(f'Sequence/request_sequences.User.txt')
        df['EditingScore'] = outputs
        TOP_N = int(TOP_N)
        best_df = df.nlargest(TOP_N, 'EditingScore')
        # best_df = best_df[['Target(47bp)', 'PBS', 'RT', 'score']]
        best_df = best_df[['Strand', 'Spacer', 'PAM', 'PBS', 'RT', 'EditToNickDistance', 'sgRNASpacer', 'sgRNAPAM',
                           'NickToNickDistance', 'EditingScore']]
        # best_df = best_df.rename(columns={'Target(47bp)': 'Target'})
        # df["EditingScore"] = outputs
        # df.to_csv(f"pegsub/static/result_update.txt", sep="\t")
    print(f'Predicting time: {time.time() - start}')
    return best_df


def transformer_predictor_order3_file_pos(transformer, X_test, batch_size_test, device, TOP_N, df):
    # test
    transformer.eval()
    n = len(X_test)
    batch_num = n // batch_size_test + 1
    print("batch_num", batch_num)
    start = time.time()
    with torch.no_grad():
        outputs = []
        att_weights = []
        best = {'max': 0, 'Target': 'ACGT', 'PBS': 'ACGT', 'RT': 'ACGT'}
        for i in range(batch_num):
            start_i = i * batch_size_test
            end_i = start_i + batch_size_test
            xb = X_test.iloc[start_i:end_i, :]

            input = (torch.tensor(list(xb["Target"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o2"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["Target_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["PBS_o3"]), device=device, dtype=torch.long),
                     torch.tensor(list(xb["RT_o3"]), device=device, dtype=torch.long))
            # torch.tensor(list(xb["Other"]), device=device, dtype=torch.float32))
            output_b, att_weights_b = transformer(input)
            output_b = output_b.squeeze(-1).cpu().numpy().tolist()
            # print("att_weights", att_weights_b)

            outputs = outputs + output_b
            # print(outputs)
            # df["EditingScore"] = outputs
            # df.to_csv(f'Sequence/result.txt', sep="\t")
            if not att_weights:
                att_weights = att_weights_b
            else:
                for j in range(len(att_weights_b)):
                    att_weights[j] = torch.cat((att_weights[j], att_weights_b[j]), 0)
        # print(outputs)
        # df = pd.read_table(f'Sequence/pegRNA_pos.User.txt', header=0)
        # df = pd.read_table(f'Position/request_sequences.User.txt')
        # print("ds")
        # print(output_b)
        df['EditingScore'] = outputs
        print(TOP_N)
        TOP_N = int(TOP_N)
        best_df = df.nlargest(TOP_N, 'EditingScore')
        # best_df = best_df[['Target(47bp)', 'PBS', 'RT', 'score']]
        # best_df = best_df.rename(columns={'Target(47bp)': 'Target'})
        best_df = best_df[['Strand', 'Spacer', 'PAM', 'PBS', 'RT', 'EditToNickDistance', 'sgRNASpacer', 'sgRNAPAM',
                           'NickToNickDistance', 'EditingScore']]
        # print("best_df")
        # print(best_df)
        # print(outputs)
        # df["EditingScore"] = outputs
        # df.to_csv(f"pegsub/static/result_pos.txt", sep="\t")

    print(f'Predicting time: {time.time() - start}')
    return best_df
