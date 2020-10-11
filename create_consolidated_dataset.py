# import tensorflow_datasets as tfds
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')

stopwords = stopwords.words('english')
stopwords.remove('not')


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


# df_gen_1 = pd.read_csv('forum_content.csv', names=["link", "content", "label"])
# df_gen_2 = pd.read_csv('forum_content_gen.csv', names=["link", "content", "label"])
# df_am_1 = pd.read_csv('forum_content_literature_am.csv', names=["link", "content", "label"])
# df_am_2 = pd.read_csv('forum_content_am.csv', names=["link", "content", "label"])
#
# df_gen_1['forum_type'] = "GENERAL_FORUM"
# df_gen_2['forum_type'] = "GENERAL_FORUM"
# df_am_1['forum_type'] = "AFRICAN_AMERICAN_GENERAL_FORUM"
# df_am_2['forum_type'] = "AFRICAN_AMERICAN_GENERAL_FORUM"

df_af_am_forum = pd.read_csv('content_am_new_debug.csv', names=["member_no","content","threadUrl","title","postDate","is_am","author"])
# df_wm_forum = pd.read_csv('content_gen.csv', names=["member_no","content","threadUrl","title","postDate","is_am","author"])

df_af_am_forum = df_af_am_forum[["threadUrl", "content", "is_am"]]
# df_wm_forum = df_wm_forum[["threadUrl", "content", "is_am"]]

df_af_am_forum.rename(columns={'threadUrl': 'link', 'is_am': 'label'}, inplace=True)
# df_wm_forum.rename(columns={'threadUrl': 'link', 'is_am': 'label'}, inplace=True)
df_af_am_forum['forum_type'] = "AFRICAN_AMERICAN_CANCER_FORUM"
# df_wm_forum['forum_type'] = "GENERAL_CANCER_FORUM"

# all_df = pd.concat([df_gen_1, df_gen_2, df_am_1, df_am_2, df_af_am_forum, df_wm_forum], ignore_index=True)
all_df = pd.concat([df_af_am_forum], ignore_index=True)
print(all_df)
print(all_df.isna().sum())

all_df.reset_index(inplace=True, drop=True)
all_df.dropna(inplace=True)

all_df['content'] = all_df['content'].str.lower()


def tokenizer(x):
    text_tokens = x.split()
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    return (" ").join(tokens_without_sw)

def pre_process(x):
    if x is None or x =='':
        return x
    x = x.strip()
    if "said:" in x:
        return x.split("said:", 1)[1].strip()

    if "wrote:" in x:
        return x.split("wrote:", 1)[1].strip()

    return x

all_df['content'] = all_df['content'].apply(pre_process)
all_df['content_processed'] = all_df['content'].apply(tokenizer)

all_df.to_csv('processed_am_all_data_debug.csv', index=False)

print(all_df.isna().sum())

#
# # string_tst = "Apr 2, 2019 01:12PM LoveFromPhilly wrote: yes yes about the gathering information!! It helps me!!! And I’m still a wreck with scans. I thought last time I was cool and then ended up running to the bathroom with nervous diarrhea right before the scan...then I realize I’m foolin myself and then I buckle down and take a Valium. Helps so much!! I wish my mind were stronger to be able to deal with them but it just ain’t. I simply cannot handle the anxiety. It’s quite rough on my mind and body - and if I’m trying to cut back on stress, this is def one I’m happy to cut back on! When waiting for the results and scans, I can tell myself I am fine but truly I am not fine in my head - I’m losing it inside! BUT yes telling myself that it’s an information gathering stage somehow does really help. That and also now the experience of realizing that sometimes scan results are incorrect so Not to get too worked up about any one thing until all that info is gathered up by my MO and then discussed with his Onco board of docs and then brought back to me. Hard not to freak out during this time... Another thing that helps me is that (and I don’t know if this is accurate but it feels this way in my brain) I try to remember that nothing bananas is going to happen overnight. This is a slow process and needs lots of angles covered - because cancer is so very multifaceted. There will be some breathing room for making choices and decisions if necessary. Big hugs abound!!!"
