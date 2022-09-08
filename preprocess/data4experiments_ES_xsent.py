import os
from collections import defaultdict
from itertools import groupby


def prepare4conversion(sentence):

    filename, sent_id, token_list = sentence

    same_sentence_event = defaultdict(list)
    event_id_token = {}
    candidate_tlink = {}
    data = []

    for elem in token_list:
        sent_id = elem.strip().split("\t")[0]
        token_sent_id = elem.strip().split("\t")[1]
        token = elem.strip().split("\t")[2]
        event_id = elem.strip().split("\t")[3]
#        timex = elem.strip().split("\t")[2]
        tlink = elem.strip().split("\t")[4]
        data.append((token_sent_id, token, event_id))

        if event_id.startswith("e"):
            same_sentence_event[sent_id].append(event_id)
            event_id_token[event_id] = token
        if tlink != "_":
           tlink_splitted = tlink.split("|")
           for rel in tlink_splitted:

                source = rel.split(":")[0]
                target = rel.split(":")[1]
                rel_type = rel.split(":")[2]

                if source.startswith("e") and target.startswith("e"):
                    candidate_tlink[(source, target)] = rel_type

    same_sentence_tlink = {}
    for events, rel_type in candidate_tlink.items():
        source, target = events
        for sentence, event_list in same_sentence_event.items():
            if source in event_list and target in event_list:
                same_sentence_tlink[(source, target, rel_type)] = data

    tlink_source = {}
    for events, sentence_data in same_sentence_tlink.items():
        source, target, value = events
        for idx, elem in enumerate(sentence_data):
            id_token, token, event_id = elem
            if source == event_id:
                # print(sentence_data, source, idx)
                tlink_source[events + (idx,) + (token,)] = sentence_data

    tlink_final = {}

    for events, sentence_data in tlink_source.items():
        source, target, value, idx_source, source_token = events
        for idx, elem in enumerate(sentence_data):
            id_token, token, event_id = elem
            if target == event_id:
                # print(sentence_data, source, idx)
                tlink_final[events + (idx,) + (token,)] = sentence_data

    return tlink_final


# [i[0] for i in a]


def process_data(input_file):

    counter = -1
    outfile = "./TB_ES_col/" + "test.txt"
    output = open(outfile, 'a', encoding='utf-8')

    with open(input_file, encoding='latin-1') as f:
        grps = groupby(f, key=lambda line: bool(line.strip()))
        for k, v in grps:
            if k:
                counter += 1
                # sentence_[(input_file, counter,)] = list(v)
                sentence = (input_file, counter, list(v),) # tupla
                tlink2print = prepare4conversion(sentence)
                if len(tlink2print) > 0:


                    for k, v in tlink2print.items():
                        source_event, target_event, tlink, source_idx, source_token, target_index, target_token = k
                        textual_data = [i[1] for i in v]
                    #        print(textual_data, k)
                        #print(" ".join(textual_data))
                        #print(" ".join(textual_data) + "\t" + str(source_idx) + "\t" + source_token + "\t" + str(target_index) + "\t" + target_token + "\t" + tlink)
                        output.writelines(" ".join(textual_data) + "\t" + str(source_idx) + "\t" + source_token + "\t" + str(target_index) + "\t" + target_token + "\t" + tlink + "\n")
    output.close()


def get_data(inputdir):
    for f in os.listdir(inputdir):
        process_data(inputdir + f)


if __name__ == '__main__':

    inputtrain_dir = "/home/p281734/projects/tlink_probing/workspace/TB_ES_col/test/"
    get_data(inputtrain_dir)