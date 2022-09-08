import sys, os, re
from collections import defaultdict, OrderedDict



def merge_token_event_labels(tokenf, eventf):

    tokens_file = defaultdict(list)

    with open(tokenf, encoding='latin1') as f:
        for line in f:
            line_stripped = line.strip()
            line_splitted =line_stripped.split("\t")
            tokens_file[(line_splitted[0], line_splitted[1], line_splitted[2])].append(line_splitted[3])

    token_events = {}
    with open(eventf, encoding='latin1') as f1:
        for line in f1:
            line_stripped = line.strip()
            line_splitted = line_stripped.split("\t")
            key_match = (line_splitted[0], line_splitted[1], line_splitted[2])

            if key_match in tokens_file:
                #print(line_splitted[4])
                tokens_file[key_match].append(line_splitted[4])
                token_events[key_match] = tokens_file[key_match]

    for k, v in tokens_file.items():
        if k not in token_events:
            v.append("_")
            token_events[k] = v

#    print(token_events)

    return token_events

def merge_tlinks_events(main_events, sub_events):

    tlink_events = defaultdict(list)

    with open(main_events, encoding='latin1') as f:
        for line in f:
            line_stripped = line.strip()
            line_splitted =line_stripped.split("\t")
#            print(line_splitted)
            tlink_events[(line_splitted[0], line_splitted[1])].append((line_splitted[1] + ":" + line_splitted[3] + ":" +line_splitted[-1]))

    with open(sub_events, encoding='latin1') as f:
        for line in f:
            line_stripped = line.strip()
            line_splitted =line_stripped.split("\t")
#            print(line_splitted)
            key = (line_splitted[0], line_splitted[1])
            if key in tlink_events:
                tlink_events[key].append((line_splitted[1] + ":" + line_splitted[3] + ":" +line_splitted[-1]))
                tlink_events[key] = tlink_events[key]
            else:
                tlink_events[(line_splitted[0], line_splitted[1])].append((line_splitted[1] + ":" + line_splitted[3] + ":" +line_splitted[-1]))

    return tlink_events

def merge4col(token_dict, tlink_dict):

    #token_event_sorted = OrderedDict(sorted(token_dict.items()))

    doc_tlink_full = defaultdict(list)

    #print(tlink_dict)
    for doc, tokens in token_dict.items():
        doc_id, sent, sent_tok = doc
        token = tokens[0]
        event_id = tokens[1]
        #print(tokens)

        event_tlink_match = (doc_id, event_id)
        #print(event_tlink_match)
        if event_tlink_match in tlink_dict:
            #print(token + "\t" + event_id + "\t" + "|".join(tlink_dict[event_tlink_match]))
#            doc_tlink_full[(doc_id, int(sent), int(sent_tok))] = token + "\t" + event_id + "\t" + "|".join(tlink_dict[event_tlink_match])
            doc_tlink_full[doc_id].append((int(sent), int(sent_tok), token, event_id, "|".join(tlink_dict[event_tlink_match])))

        elif (event_tlink_match not in tlink_dict and event_id != None):
#                doc_tlink_full[doc_id].append((int(sent), int(sent_tok), token, event_id, "_"))
#            doc_tlink_full[(doc_id, int(sent), int(sent_tok))] = token + "\t" + event_id + "\t" + "_"
            doc_tlink_full[doc_id].append((int(sent), int(sent_tok), token, event_id, "_"))

        else:
            #doc_tlink_full[doc_id].append((int(sent), int(sent_tok), token, "_", "_"))
#            doc_tlink_full[(doc_id, int(sent), int(sent_tok))] = token + "\t" + "_" + "\t" + "_"
            doc_tlink_full[doc_id].append((int(sent), int(sent_tok), token, "_", "_"))

    #print(doc_tlink_full)
    token_event_sorted = OrderedDict(doc_tlink_full)

#    print(doc_tlink_full)

    for doc, data in token_event_sorted.items():
        #sorted_data = sorted(data, key=lambda element: (element[1], element[2]))
        data.sort(key=lambda t: (t[0], t[1]))

        outfile = "./TB_ES/" + doc + ".col"
        output = open(outfile, 'a')
        for elem in data:
            sent_id, token_sent, token, event, tlinks = elem
            if sent_id == 1 and token_sent == 1:
                output.writelines(str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks + "\n")
        #        print(str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks)
            elif sent_id != 1 and token_sent == 1:
        #        print(str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks)
                output.writelines("\n" + str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks+ "\n")
            else:
        #        print(str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks)
                output.writelines(str(sent_id) + "\t" + str(token_sent) + "\t" + token + "\t" + event + "\t" + tlinks + "\n")

        output.close()

if __name__ == '__main__':
    inputf1 = "/home/p281734/projects/tlink_probing/TB_ES/spanish_timebank_v1/data/base-segmentation.csv"
    inputf2 = "/home/p281734/projects/tlink_probing/TB_ES/spanish_timebank_v1/data/event_extents.csv"
    inputf3 = "/home/p281734/projects/tlink_probing/TB_ES/spanish_timebank_v1/data/tlinks_main_events.csv"
    inputf4 = "/home/p281734/projects/tlink_probing/TB_ES/spanish_timebank_v1/data/tlinks_subordinated_events.csv"

    token_events  = merge_token_event_labels(inputf1,inputf2)
    tlink_events = merge_tlinks_events(inputf3,inputf4)

    merge4col(token_events, tlink_events)

