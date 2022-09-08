import os
from collections import defaultdict

def get_data(inputdir):


    for f in os.listdir(inputdir):

        candidate_tlink = {}
        sentences_tokens = defaultdict(list)
        event_sentence ={}
        same_sentence_events = defaultdict(list)

        with open(inputdir + f, encoding='latin1') as inputf:
            for line in inputf:
                line_stripped = line.strip()
                line_splitted = line_stripped.split("\t")
                if len(line_splitted) > 1:

                    sentences_tokens[line_splitted[2]].append((line_splitted[0], line_splitted[1], line_splitted[3], line_splitted[4]))
                    if line_splitted[3].startswith("e"):
                        event_sentence[line_splitted[3]] = line_splitted[2]
#                        same_sentence_events[line_splitted[2]].append(line_splitted[3])

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
        sentence_tlink = {}
        for events, rel_type in candidate_tlink.items():
            source, target = events
            if source in event_sentence and target in event_sentence:
                s1 = sentences_tokens[event_sentence[source]]
                s2 = sentences_tokens[event_sentence[target]]
                sentence_tlink[(source, event_sentence[source], target, event_sentence[target], rel_type)] = (s1,s2)

        tlink_source = {}
        for events, sentence_data in sentence_tlink.items():
            source, source_sent_id, target, target_sentence_id, value = events

            for idx, elem in enumerate(sentence_data[0]):
                token, id, event_id, event_bio = elem
                if source == event_id and event_bio.startswith("B-"):
                    #print(elem, source, idx)
                    tlink_source[events + (idx,) + (token,)] = sentence_data


        tlink_final = {}

        for events, sentence_data in tlink_source.items():
            source, source_sent_id, target, target_sentence_id, value, idx_source, source_token = events
            for idx, elem in enumerate(sentence_data[1]):
                token, id, event_id, event_bio = elem
                if target == event_id and event_bio.startswith("B-"):
#                    #print(sentence_data, source, idx)
                    tlink_final[events + (idx,) + (token,)] = sentence_data

        """
        print train/test data
        """

        #print(tlink_final)
        outfile = "./TB_EN_col/" + "train_v2.txt"
        output = open(outfile, 'a')

        for k, v in tlink_final.items():
            source_event, source_sent_id, target_event, target_sentence_id, tlink, source_idx, source_token, target_index, target_token = k
            s1_ = [i[0] for i in v[0]]
            s2_ = [i[0] for i in v[1]]
            if source_sent_id == target_sentence_id:
                if source_idx > target_index:
                    if tlink == "BEFORE":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "AFTER" + "\n")
                    elif tlink == "AFTER":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "BEFORE" + "\n")
                    elif tlink == "IS_INCLUDED":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "INCLUDES" + "\n")
                    elif tlink == "INCLUDES":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "IS_INCLUDED" + "\n")
                    elif tlink == "IBEFORE":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "IAFTER" + "\n")
                    elif tlink == "IAFTER":
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + "IBEFORE" + "\n")
                    else:
                        output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(target_index) + "\t" + target_token + "\t" +  "s2:" +  str(source_idx) + "\t" + source_token + "\t" + tlink + "\n")

            else:
                output.writelines(" ".join(s1_) + "\t" + " ".join(s2_) + "\t" + "s1:" + str(source_idx) + "\t" + source_token + "\t" + "s2:" + str(target_index) + "\t" + target_token + "\t" + tlink + "\n")


        output.close()


#[i[0] for i in a]




if __name__ == '__main__':

    inputtrain_dir = "/home/p281734/projects/tlink_probing/workspace/TB_EN_col/train/"

    get_data(inputtrain_dir)