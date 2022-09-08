import os
from collections import defaultdict

def get_data(inputdir):


    for f in os.listdir(inputdir):

        candidate_tlink = {}
        sentences_tokens = defaultdict(list)
        same_sentence_events = defaultdict(list)

        with open(inputdir + f, encoding='latin1') as inputf:
            for line in inputf:
                line_stripped = line.strip()
                line_splitted = line_stripped.split("\t")
                if len(line_splitted) > 1:

                    sentences_tokens[line_splitted[2]].append((line_splitted[0], line_splitted[1], line_splitted[3], line_splitted[4]))
                    if line_splitted[3].startswith("e"):
                        same_sentence_events[line_splitted[2]].append(line_splitted[3])

                    tlink = line_splitted[17]
                    if "||" in tlink:
                        tlink_splitted = tlink.split("||")
                        for elem in tlink_splitted:
                            source = elem.split(":")[0]
                            target = elem.split(":")[1]
                            rel_type = elem.split(":")[2]
                            if source.startswith("e") and target.startswith("e"):
                                #candidate_tlink.append((source,target,rel_type))
                                candidate_tlink[(source,target)] = rel_type


        """
        merge list and prepare data for output
        """

#        for sentence, token_list in sentences_tokens.items():
#            for elem in token_list:
#                token, unique_id, event_status = elem
#                if event_status in same_sentence_events[sentence]:
#                    print(elem)
        same_sentence_tlink = {}
        for events, rel_type in candidate_tlink.items():
            source, target = events
            for sentence, event_list in same_sentence_events.items():
                if source in event_list and target in event_list:
                    data = sentences_tokens[sentence]
                    same_sentence_tlink[(source, target, rel_type)] = data
                    #print(sentence, source, target, rel_type, data)

        tlink_source = {}
        for events, sentence_data in same_sentence_tlink.items():
            source, target, value = events
            for idx, elem in enumerate(sentence_data):
                token, id, event_id, event_bio = elem
                if source == event_id and event_bio.startswith("B-"):
                    #print(sentence_data, source, idx)
                    tlink_source[events + (idx,) + (token,)] = sentence_data

        tlink_final = {}

        for events, sentence_data in tlink_source.items():
            source, target, value, idx_source, source_token = events
            for idx, elem in enumerate(sentence_data):
                token, id, event_id, event_bio = elem
                if target == event_id and event_bio.startswith("B-"):
                    #print(sentence_data, source, idx)
                    tlink_final[events + (idx,) + (token,)] = sentence_data

        """
        print train/test data
        """

        #print(tlink_final)
        outfile = "./TB_FR_col/" + "full_data.txt"
        output = open(outfile, 'a')

        for k, v in tlink_final.items():
            source_event, target_event, tlink, source_idx, source_token, target_index, target_token = k
            textual_data = [i[0] for i in v]
            output.writelines(" ".join(textual_data) + "\t" + str(source_idx) + "\t" + source_token + "\t" + str(target_index) + "\t" + target_token + "\t" + tlink + "\n")
        output.close()


#[i[0] for i in a]




if __name__ == '__main__':

    inputtrain_dir = "/home/p281734/projects/tlink_probing/workspace/TB_FR_col/"

    get_data(inputtrain_dir)