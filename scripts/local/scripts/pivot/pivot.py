# for every row in big_corpus_en.csv 

# compute the translation of the en sentence into zh using DeepL 

# add this to the pivot zh corpus + the original translations from formosan <-> zh 

# for every row in big_corpus_zh.csv

# compute the translation of the zh sentence into en using DeepL

# add this to the pivot en corpus + the original translations from formosan <-> en

# now we have two pivot corpora 

# RESPECT THE ORIGINAL SPLITS / STRUCTURE 

# EXAMPLE STRUCTURE: big_corpus_en.csv

lang_code,formosan_sentence,english_sentence,source,dialect,split
ami,cacay,one,Formosan-ePark/Final_XML/xue_xi_ci_biao_learning_vocabulary/Amis/Southern_Amis.xml,Southern,train

# EXAMPLE STRUCTURE: big_corpus_zh.csv

lang_code,formosan_sentence,chinese_sentence,source,dialect,split
ami,kamaya.,這個紅柿子有毛。,Formosan-Zheng-Data/Final_XML/Amis/Parallel_Amis_ami.xml,UNKNOWN,train


the Deepl api key is in the .env as DEEPL_API_KEY=....key