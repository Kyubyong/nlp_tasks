# Natural Language Processing Tasks and Selected References

I've been working on several natural language processing tasks for a long time. One day, I felt like drawing a map of the NLP field where I earn a living. I'm sure I'm not the only person who wants to see at a glance which tasks are in NLP.

I did my best to cover as many as possible tasks in NLP, but admittedly this is far from exhaustive purely due to my lack of knowledge. And selected references are biased towards recent deep learning accomplishments. I expect these serve as a starting point when you're about to dig into the task. I'll keep updating this repo myself, but what I really hope is you collaborate on this work. Don't hesitate to send me a pull request!

Oct. 13, 2017.<br/>
by Kyubyong

Reviewed and updated by [YJ Choe](https://github.com/yjchoe) on Oct. 18, 2017.

## Anaphora Resolution
  * See [Coreference Resolution](#coreference-resolution)

## Automated Essay Scoring
  * ****`PAPER`**** [Automatic Text Scoring Using Neural Networks](https://arxiv.org/abs/1606.04289)
  * ****`PAPER`**** [A Neural Approach to Automated Essay Scoring](http://www.aclweb.org/old_anthology/D/D16/D16-1193.pdf)
  * ****`CHALLENGE`**** [Kaggle: The Hewlett Foundation: Automated Essay Scoring](https://www.kaggle.com/c/asap-aes)
  * ****`PROJECT`**** [EASE (Enhanced AI Scoring Engine)](https://github.com/edx/ease)

## Automatic Speech Recognition
  * ****`WIKI`**** [Speech recognition](https://en.wikipedia.org/wiki/Speech_recognition)
  * ****`PAPER`**** [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
  * ****`PAPER`**** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
  * ****`PROJECT`**** [A TensorFlow implementation of Baidu's DeepSpeech architecture](https://github.com/mozilla/DeepSpeech)
  * ****`PROJECT`**** [Speech-to-Text-WaveNet : End-to-end sentence level English speech recognition using DeepMind's WaveNet](https://github.com/buriburisuri/speech-to-text-wavenet)
  * ****`CHALLENGE`**** [The 5th CHiME Speech Separation and Recognition Challenge](http://spandh.dcs.shef.ac.uk/chime_challenge/)
  * ****`DATA`**** [The 5th CHiME Speech Separation and Recognition Challenge](http://spandh.dcs.shef.ac.uk/chime_challenge/download.html)
  * ****`DATA`**** [CSTR VCTK Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
  * ****`DATA`**** [LibriSpeech ASR corpus](http://www.openslr.org/12/)
  * ****`DATA`**** [Switchboard-1 Telephone Speech Corpus](https://catalog.ldc.upenn.edu/ldc97s62)
  * ****`DATA`**** [TED-LIUM Corpus](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)
  * ****`DATA`**** [Open Speech and Language Resources](http://www.openslr.org/)
  * ****`DATA`**** [Common Voice](https://voice.mozilla.org/en/data)

## Automatic Summarisation
  * ****`WIKI`**** [Automatic summarization](https://en.wikipedia.org/wiki/Automatic_summarization)
  * ****`BOOK`**** [Automatic Text Summarization](https://www.amazon.com/Automatic-Text-Summarization-Juan-Manuel-Torres-Moreno/dp/1848216688/ref=sr_1_1?s=books&ie=UTF8&qid=1507782304&sr=1-1&keywords=Automatic+Text+Summarization)
  * ****`PAPER`**** [Text Summarization Using Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.823.8025&rep=rep1&type=pdf)
  * ****`PAPER`**** [Ranking with Recursive Neural Networks and Its Application to Multi-Document Summarization](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9414/9520)
  * ****`DATA`**** [Text Analytics Conferences (TAC)](https://tac.nist.gov/data/index.html)
  * ****`DATA`**** [Document Understanding Conferences (DUC)](http://www-nlpir.nist.gov/projects/duc/data.html)

## Coreference Resolution
  * ****`INFO`**** [Coreference Resolution](https://nlp.stanford.edu/projects/coref.shtml)
  * ****`PAPER`**** [Deep Reinforcement Learning for Mention-Ranking Coreference Models](https://arxiv.org/abs/1609.08667)
  * ****`PAPER`**** [Improving Coreference Resolution by Learning Entity-Level Distributed Representations](https://arxiv.org/abs/1606.01323)
  * ****`CHALLENGE`**** [CoNLL 2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes](http://conll.cemantix.org/2012/task-description.html)
  * ****`CHALLENGE`**** [CoNLL 2011 Shared Task: Modeling Unrestricted Coreference in OntoNotes](http://conll.cemantix.org/2011/task-description.html)
  * ****`CHALLENGE`**** [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](https://competitions.codalab.org/competitions/17310)
  
## Entity Linking
  * See [Named Entity Disambiguation](#named-entity-disambiguation)

## Grammatical Error Correction
  * ****`PAPER`**** [Neural Network Translation Models for Grammatical Error Correction](https://arxiv.org/abs/1606.00189)
  * ****`PAPER`**** [Adapting Sequence Models for Sentence Correction](http://aclweb.org/anthology/D17-1297)
  * ****`CHALLENGE`**** [CoNLL-2013 Shared Task: Grammatical Error Correction](http://www.comp.nus.edu.sg/~nlp/conll13st.html)
  * ****`CHALLENGE`**** [CoNLL-2014 Shared Task: Grammatical Error Correction](http://www.comp.nus.edu.sg/~nlp/conll14st.html)
  * ****`DATA`**** [NUS Non-commercial research/trial corpus license](http://www.comp.nus.edu.sg/~nlp/conll14st/nucle_license.pdf)
  * ****`DATA`**** [Lang-8 Learner Corpora](http://cl.naist.jp/nldata/lang-8/)
  * ****`DATA`**** [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/%7Ecristian/Cornell_Movie-Dialogs_Corpus.html)
  * ****`PROJECT`**** [Deep Text Corrector](https://github.com/atpaino/deep-text-corrector)
  * ****`PRODUCT`**** [deep grammar](http://deepgrammar.com/)

## Grapheme To Phoneme Conversion
  * ****`PAPER`**** [Grapheme-to-Phoneme Models for (Almost) Any Language](https://pdfs.semanticscholar.org/b9c8/fef9b6f16b92c6859f6106524fdb053e9577.pdf)
  * ****`PAPER`**** [Polyglot Neural Language Models: A Case Study in Cross-Lingual Phonetic Representation Learning](https://arxiv.org/pdf/1605.03832.pdf)
  * ****`PAPER`**** [Multitask Sequence-to-Sequence Models for Grapheme-to-Phoneme Conversion](https://pdfs.semanticscholar.org/26d0/09959fa2b2e18cddb5783493738a1c1ede2f.pdf)
  * ****`PROJECT`**** [Sequence-to-Sequence G2P toolkit](https://github.com/cmusphinx/g2p-seq2seq)
  * ****`DATA`**** [Multilingual Pronunciation Data](https://drive.google.com/drive/folders/0B7R_gATfZJ2aWkpSWHpXUklWUmM)

## Humor and Sarcasm Detection
  * ****`PAPER`**** [Automatic Sarcasm Detection: A Survey](https://arxiv.org/abs/1602.03426)
  * ****`PAPER`**** [Magnets for Sarcasm: Making Sarcasm Detection Timely, Contextual and Very Personal](http://aclweb.org/anthology/D17-1051)
  * ****`PAPER`**** [Sarcasm Detection on Twitter: A Behavioral Modeling Approach](http://ai2-s2-pdfs.s3.amazonaws.com/67b5/9db00c29152d8e738f693f153e1ab9b43466.pdf)
  * ****`CHALLENGE`**** [SemEval-2017 Task 6: #HashtagWars: Learning a Sense of Humor](http://alt.qcri.org/semeval2017/task6/)
  * ****`CHALLENGE`**** [SemEval-2017 Task 7: Detection and Interpretation of English Puns](http://alt.qcri.org/semeval2017/task7/)
  * ****`DATA`**** [Sarcastic comments from Reddit](https://www.kaggle.com/danofer/sarcasm/)
  * ****`DATA`**** [Sarcasm Corpus V2](https://nlds.soe.ucsc.edu/sarcasm2)
  * ****`DATA`**** [Sarcasm Amazon Reviews Corpus](https://github.com/ef2020/SarcasmAmazonReviewsCorpus)

## Language Grounding
  * ****`WIKI`**** [Symbol grounding problem](https://en.wikipedia.org/wiki/Symbol_grounding_problem)
  * ****`PAPER`**** [The Symbol Grounding Problem](http://courses.media.mit.edu/2004spring/mas966/Harnad%20symbol%20grounding.pdf)
  * ****`PAPER`**** [From phonemes to images: levels of representation in a recurrent neural model of visually-grounded language learning](https://arxiv.org/abs/1610.03342)
  * ****`PAPER`**** [Encoding of phonology in a recurrent neural model of grounded speech](https://arxiv.org/abs/1706.03815)
  * ****`PAPER`**** [Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230)
  * ****`PAPER`**** [Sound-Word2Vec: Learning Word Representations Grounded in Sounds](https://arxiv.org/abs/1703.01720)
  * ****`COURSE`**** [Language Grounding to Vision and Control](https://www.cs.cmu.edu/~katef/808/)
  * ****`WORKSHOP`**** [Language Grounding for Robotics](https://robonlp2017.github.io/)

## Language Guessing
  * See [Language Identification](#language-identification)

## Language Identification
  * ****`WIKI`**** [Language identification](https://en.wikipedia.org/wiki/Language_identification)
  * ****`PAPER`**** [AUTOMATIC LANGUAGE IDENTIFICATION USING DEEP NEURAL NETWORKS](https://repositorio.uam.es/bitstream/handle/10486/666848/automatic_lopez-moreno_ICASSP_2014_ps.pdf?sequence=1)
  * ****`PAPER`**** [Natural Language Processing with Small Feed-Forward Networks](http://aclweb.org/anthology/D17-1308)
  * ****`CHALLENGE`**** [2015 Language Recognition Evaluation](https://www.nist.gov/itl/iad/mig/2015-language-recognition-evaluation)

## Language Modeling
  * ****`WIKI`**** [Language model](https://en.wikipedia.org/wiki/Language_model)
  * ****`TOOLKIT`**** [KenLM Language Model Toolkit](http://kheafield.com/code/kenlm/)
  * ****`PAPER`**** [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  * ****`PAPER`**** [Generating Sequences with Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
  * ****`PAPER`**** [Character-Aware Neural Language Models](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017)
  * ****`THESIS`**** [Statistical Language Models Based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
  * ****`DATA`**** [Penn Treebank](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data)
  * ****`TUTORIAL`**** [TensorFlow Tutorial on Language Modeling with Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent#language_modeling)

## Language Recognition
  * See [Language Identification](#language-identification)

## Lemmatisation
  * ****`WIKI`**** [Lemmatisation](https://en.wikipedia.org/wiki/Lemmatisation)
  * ****`PAPER`**** [Joint Lemmatization and Morphological Tagging with LEMMING](http://www.cis.lmu.de/~muellets/pdf/emnlp_2015.pdf)
  * ****`TOOLKIT`**** [WordNet Lemmatizer](http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize)
  * ****`DATA`**** [Treebank-3](https://catalog.ldc.upenn.edu/ldc99t42)

## Lip-reading
  * ****`WIKI`**** [Lip reading](https://en.wikipedia.org/wiki/Lip_reading)
  * ****`PAPER`**** [Lip Reading Sentences in the Wild](https://arxiv.org/abs/1611.05358)
  * ****`PAPER`**** [3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition](https://arxiv.org/abs/1706.05739)
  * ****`PROJECT`**** [Lip Reading - Cross Audio-Visual Recognition using 3D Convolutional Neural Networks](https://github.com/astorfi/lip-reading-deeplearning)
  * ****`DATA`**** [The GRID audiovisual sentence corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)
  * ****`DATA`**** [The BBC-Oxford 'Multi-View Lip Reading Sentences' (MV-LRS) Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading_sentences/)

## Machine Translation
  * ****`PAPER`**** [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
  * ****`PAPER`**** [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)
  * ****`PAPER`**** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * ****`PAPER`**** [Six Challenges for Neural Machine Translation](http://aclweb.org/anthology/W/W17/W17-3204.pdf)
  * ****`CHALLENGE`**** [ACL 2014 NINTH WORKSHOP ON STATISTICAL MACHINE TRANSLATION](http://www.statmt.org/wmt14/translation-task.html#download)
  * ****`CHALLENGE`**** [EMNLP 2017 SECOND CONFERENCE ON MACHINE TRANSLATION (WMT17) ](http://www.statmt.org/wmt17/translation-task.html)
  * ****`DATA`**** [OpenSubtitles2016](http://opus.lingfil.uu.se/OpenSubtitles2016.php)
  * ****`DATA`**** [WIT3: Web Inventory of Transcribed and Translated Talks](https://wit3.fbk.eu/)
  * ****`DATA`**** [The QCRI Educational Domain (QED) Corpus](http://alt.qcri.org/resources/qedcorpus/)
  * ****`PAPER`**** [Multi-task Sequence to Sequence Learning](https://arxiv.org/abs/1511.06114)
  * ****`PAPER`**** [Unsupervised Pretraining for Sequence to Sequence Learning](http://aclweb.org/anthology/D17-1039)
  * ****`PAPER`**** [Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558)
  * ****`TOOLKIT`**** [Subword Neural Machine Translation with Byte Pair Encoding (BPE)](https://github.com/rsennrich/subword-nmt)
  * ****`TOOLKIT`**** [Multi-Way Neural Machine Translation](https://github.com/nyu-dl/dl4mt-multi)
  * ****`TOOLKIT`**** [OpenNMT: Open-Source Toolkit for Neural Machine Translation](http://opennmt.net/)

## Morphological Inflection Generation
  * ****`WIKI`**** [Inflection](https://en.wikipedia.org/wiki/Inflection)
  * ****`PAPER`**** [Morphological Inflection Generation Using Character Sequence to Sequence Learning](https://arxiv.org/abs/1512.06110)
  * ****`CHALLENGE`**** [SIGMORPHON 2016 Shared Task: Morphological Reinflection](http://ryancotterell.github.io/sigmorphon2016/)
  * ****`DATA`**** [sigmorphon2016](https://github.com/ryancotterell/sigmorphon2016)

## Named Entity Disambiguation
  * ****`WIKI`**** [Entity linking](https://en.wikipedia.org/wiki/Entity_linking)
  * ****`PAPER`**** [Robust and Collective Entity Disambiguation through Semantic Embeddings](http://www.stefanzwicklbauer.info/pdf/Sigir_2016.pdf)

## Named Entity Recognition
  * ****`WIKI`**** [Named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
  * ****`PAPER`**** [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
  * ****`PROJECT`**** [OSU Twitter NLP Tools](https://github.com/aritter/twitter_nlp)
  * ****`CHALLENGE`**** [Named Entity Recognition in Twitter](https://noisy-text.github.io/2016/ner-shared-task.html)
  * ****`CHALLENGE`**** [CoNLL 2002 Language-Independent Named Entity Recognition](https://www.clips.uantwerpen.be/conll2002/ner/)
  * ****`CHALLENGE`**** [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://aclweb.org/anthology/W03-0419)
  * ****`DATA`**** [CoNLL-2002 NER corpus](https://github.com/teropa/nlp/tree/master/resources/corpora/conll2002)
  * ****`DATA`**** [CoNLL-2003 NER corpus](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)
  * ****`DATA`**** [NUT Named Entity Recognition in Twitter Shared task](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16)
  * ****`TOOLKIT`**** [Stanford Named Entity Recognizer](https://nlp.stanford.edu/software/CRF-NER.shtml)

## Paraphrase Detection
  * ****`PAPER`**** [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.650.7199&rep=rep1&type=pdf)
  * ****`PROJECT`**** [Paralex: Paraphrase-Driven Learning for Open Question Answering](http://knowitall.cs.washington.edu/paralex/)
  * ****`CHALLENGE`**** [SemEval-2015 Task 1: Paraphrase and Semantic Similarity in Twitter](http://alt.qcri.org/semeval2015/task1/)
  * ****`DATA`**** [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
  * ****`DATA`**** [Microsoft Research Video Description Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52422&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F38cf15fd-b8df-477e-a4e4-a4680caa75af%2F)
  * ****`DATA`**** [Pascal Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html)
  * ****`DATA`**** [Flickr Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)
  * ****`DATA`**** [The SICK data set](http://clic.cimec.unitn.it/composes/sick.html)
  * ****`DATA`**** [PPDB: The Paraphrase Database](http://www.cis.upenn.edu/%7Eccb/ppdb/)
  * ****`DATA`**** [WikiAnswers Paraphrase Corpus](http://knowitall.cs.washington.edu/paralex/wikianswers-paraphrases-1.0.tar.gz)
  
## Paraphrase Generation  
* ****`PAPER`**** [Neural Paraphrase Generation with Stacked Residual LSTM Networks](https://arxiv.org/pdf/1610.03098.pdf)
* ****`DATA`**** [Neural Paraphrase Generation with Stacked Residual LSTM Networks](https://github.com/iamaaditya/neural-paraphrase-generation/tree/master/data)
* ****`CODE`**** [Neural Paraphrase Generation with Stacked Residual LSTM Networks](https://github.com/iamaaditya/neural-paraphrase-generation)
* ****`PAPER`**** [A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/pdf/1709.05074.pdf)
* ****`PAPER`**** [Paraphrasing Revisited with Neural Machine Translation](http://www.research.ed.ac.uk/portal/files/34902784/document.pdf)


## Parsing
  * ****`WIKI`**** [Parsing](https://en.wikipedia.org/wiki/Parsing)
  * ****`TOOLKIT`**** [The Stanford Parser: A statistical parser](https://nlp.stanford.edu/software/lex-parser.shtml)
  * ****`TOOLKIT`**** [spaCy parser](https://spacy.io/docs/usage/dependency-parse)
  * ****`PAPER`**** [Grammar as a Foreign Language](https://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf)
  * ****`PAPER`**** [A fast and accurate dependency parser using neural networks](http://www.aclweb.org/anthology/D14-1082)
  * ****`PAPER`**** [Universal Semantic Parsing](https://aclanthology.info/pdf/D/D17/D17-1009.pdf)
  * ****`CHALLENGE`**** [CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies](http://universaldependencies.org/conll17/)
  * ****`CHALLENGE`**** [CoNLL 2016 Shared Task: Multilingual Shallow Discourse Parsing](http://www.cs.brandeis.edu/~clp/conll16st/)
  * ****`CHALLENGE`**** [CoNLL 2015 Shared Task: Shallow Discourse Parsing](http://www.cs.brandeis.edu/~clp/conll15st/)
  * ****`CHALLENGE`**** [SemEval-2016 Task 8: The meaning representations may be abstract, but this task is concrete!](http://alt.qcri.org/semeval2016/task8/)

## Part-of-speech Tagging
  * ****`WIKI`**** [Part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
  * ****`PAPER`**** [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/pdf/1604.05529.pdf)
  * ****`PAPER`**** [Unsupervised Part-Of-Speech Tagging with Anchor Hidden Markov Models](https://transacl.org/ojs/index.php/tacl/article/viewFile/837/192)
  * ****`DATA`**** [Treebank-3](https://catalog.ldc.upenn.edu/ldc99t42)
  * ****`TOOLKIT`**** [nltk.tag package](http://www.nltk.org/api/nltk.tag.html)

## Pinyin-To-Chinese Conversion
  * ****`WIKI`**** [Pinyin input method](https://en.wikipedia.org/wiki/Pinyin_input_method)
  * ****`PAPER`**** [Neural Network Language Model for Chinese Pinyin Input Method Engine](http://aclweb.org/anthology/Y15-1052)
  * ****`PROJECT`**** [Neural Chinese Transliterator](https://github.com/Kyubyong/neural_chinese_transliterator)

## Question Answering
  * ****`WIKI`**** [Question answering](https://en.wikipedia.org/wiki/Question_answering)
  * ****`PAPER`**** [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://www.thespermwhale.com/jaseweston/ram/papers/paper_21.pdf)
  * ****`PAPER`**** [Dynamic Memory Networks for Visual and Textual Question Answering](http://proceedings.mlr.press/v48/xiong16.pdf)
  * ****`CHALLENGE`**** [TREC Question Answering Task](http://trec.nist.gov/data/qamain.html)
  * ****`CHALLENGE`**** [NTCIR-8: Advanced Cross-lingual Information Access (ACLIA)](http://aclia.lti.cs.cmu.edu/ntcir8/Home)
  * ****`CHALLENGE`**** [CLEF Question Answering Track](http://nlp.uned.es/clef-qa/)
  * ****`CHALLENGE`**** [SemEval-2017 Task 3: Community Question Answering](http://alt.qcri.org/semeval2017/task3/)
  * ****`CHALLENGE`**** [SemEval-2018 Task 11: Machine Comprehension using Commonsense Knowledge](https://competitions.codalab.org/competitions/17184)
  * ****`DATA`**** [MS MARCO: Microsoft MAchine Reading COmprehension Dataset](http://www.msmarco.org/)
  * ****`DATA`**** [Maluuba NewsQA](https://github.com/Maluuba/newsqa)
  * ****`DATA`**** [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://rajpurkar.github.io/SQuAD-explorer/)
  * ****`DATA`**** [GraphQuestions: A Characteristic-rich Question Answering Dataset](https://github.com/ysu1989/GraphQuestions)
  * ****`DATA`**** [Story Cloze Test and ROCStories Corpora](http://cs.rochester.edu/nlp/rocstories/)
  * ****`DATA`**** [Microsoft Research WikiQA Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F4495da01-db8c-4041-a7f6-7984a4f6a905%2Fdefault.aspx)
  * ****`DATA`**** [DeepMind Q&A Dataset](http://cs.nyu.edu/%7Ekcho/DMQA/)
  * ****`DATA`**** [QASent](http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz)
  * ****`DATA`**** [Textbook Question Answering](http://textbookqa.org/)

## Relationship Extraction
  * ****`WIKI`**** [Relationship extraction](https://en.wikipedia.org/wiki/Relationship_extraction)
  * ****`PAPER`**** [A deep learning approach for relationship extraction from interaction context in social manufacturing paradigm](http://www.sciencedirect.com/science/article/pii/S0950705116001210)
  * ****`CHALLENGE`**** [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)

## Semantic Role Labeling
  * ****`WIKI`**** [Semantic role labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling)
  * ****`BOOK`**** [Semantic Role Labeling](https://www.amazon.com/Semantic-Labeling-Synthesis-Lectures-Technologies/dp/1598298313/ref=sr_1_1?s=books&ie=UTF8&qid=1507776173&sr=1-1&keywords=Semantic+Role+Labeling)
  * ****`PAPER`**** [End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks](http://www.aclweb.org/anthology/P/P15/P15-1109.pdf)
  * ****`PAPER`**** [Neural Semantic Role Labeling with Dependency Path Embeddings](https://arxiv.org/abs/1605.07515)
  * ****`PAPER`**** [Deep Semantic Role Labeling: What Works and What's Next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)
  * ****`CHALLENGE`**** [CoNLL-2005 Shared Task: Semantic Role Labeling](http://www.cs.upc.edu/~srlconll/st05/st05.html)
  * ****`CHALLENGE`**** [CoNLL-2004 Shared Task: Semantic Role Labeling](http://www.cs.upc.edu/~srlconll/st04/st04.html)
  * ****`TOOLKIT`**** [Illinois Semantic Role Labeler (SRL)](http://cogcomp.org/page/software_view/SRL)
  * ****`DATA`**** [CoNLL-2005 Shared Task: Semantic Role Labeling](http://www.cs.upc.edu/~srlconll/soft.html)

## Sentence Boundary Disambiguation
  * ****`WIKI`**** [Sentence boundary disambiguation](https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation)
  * ****`PAPER`**** [A Quantitative and Qualitative Evaluation of Sentence Boundary Detection for the Clinical Domain](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001746/)
  * ****`TOOLKIT`**** [NLTK Tokenizers](http://www.nltk.org/_modules/nltk/tokenize.html)
  * ****`DATA`**** [The British National Corpus](http://www.natcorp.ox.ac.uk/)
  * ****`DATA`**** [Switchboard-1 Telephone Speech Corpus](https://catalog.ldc.upenn.edu/ldc97s62)

## Sentiment Analysis
  * ****`WIKI`**** [Sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
  * ****`INFO`**** [Awesome Sentiment Analysis](https://github.com/xiamx/awesome-sentiment-analysis)
  * ****`CHALLENGE`**** [Kaggle: UMICH SI650 - Sentiment Classification](https://www.kaggle.com/c/si650winter11#description)
  * ****`CHALLENGE`**** [SemEval-2017 Task 4: Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2017/task4/)
  * ****`CHALLENGE`**** [SemEval-2017 Task 5: Fine-Grained Sentiment Analysis on Financial Microblogs and News](http://alt.qcri.org/semeval2017/task5/)
  * ****`PROJECT`**** [SenticNet](http://sentic.net/about/)
  * ****`PROJECT`**** [Stanford NLP Group Sentiment Analysis](https://nlp.stanford.edu/sentiment/)
  * ****`DATA`**** [Multi-Domain Sentiment Dataset (version 2.0)](http://www.cs.jhu.edu/%7Emdredze/datasets/sentiment/)
  * ****`DATA`**** [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/code.html)
  * ****`DATA`**** [Twitter Sentiment Corpus](http://www.sananalytics.com/lab/twitter-sentiment/)
  * ****`DATA`**** [Twitter Sentiment Analysis Training Corpus](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
  * ****`DATA`**** [AFINN: List of English words rated for valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)

## Sign Language Recognition
  * ****`PAPER`**** [Video-based Sign Language Recognition without Temporal Segmentation](https://arxiv.org/pdf/1801.10111.pdf)
  * ****`PAPER`**** [SubUNets: End-to-end Hand Shape and Continuous Sign Language Recognition](http://openaccess.thecvf.com/content_ICCV_2017/papers/Camgoz_SubUNets_End-To-End_Hand_ICCV_2017_paper.pdf)  
  * ****`DATA`**** [RWTH-PHOENIX-Weather](https://www-i6.informatik.rwth-aachen.de/~forster/database-rwth-phoenix.php)
  * ****`DATA`**** [ASLLRP](http://www.bu.edu/asllrp/)

## Singing Voice Synthesis
  * ****`PAPER`**** [Singing voice synthesis based on deep neural networks](https://pdfs.semanticscholar.org/9a8e/b69480eead85f32ee4b92fa2563dd5f83401.pdf)
  * ****`PAPER`**** [A Neural Parametric Singing Synthesizer Modeling Timbre and Expression from Natural Songs](http://www.mdpi.com/2076-3417/7/12/1313)  
  * ****`PRODUCT`**** [VOCALOID: voice synthesis technology and software developed by Yamaha](https://www.vocaloid.com/en)
  * ****`CHALLENGE`**** [Special Session Interspeech 2016 Singing synthesis challenge "Fill-in the Gap"](https://chanter.limsi.fr/doku.php?id=description:start)

## Social Science Applications
  * ****`WORKSHOP`**** [NLP+CSS: Workshops on Natural Language Processing and Computational Social Science](https://sites.google.com/site/nlpandcss/)
  * ****`TOOLKIT`**** [Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints](https://github.com/uclanlp/reducingbias)  
  * ****`TOOLKIT`**** [Online Variational Bayes for Latent Dirichlet Allocation (LDA)](https://github.com/blei-lab/onlineldavb)
  * ****`GROUP`**** [The University of Chicago Knowledge Lab](http://www.knowledgelab.org/)

## Source Separation
  * ****`WIKI`**** [Source separation](https://en.wikipedia.org/wiki/Source_separation)
  * ****`PAPER`**** [From Blind to Guided Audio Source Separation](https://hal-univ-rennes1.archives-ouvertes.fr/hal-00922378/document)
  * ****`PAPER`**** [Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation](https://arxiv.org/abs/1502.04149)
  * ****`CHALLENGE`**** [Signal Separation Evaluation Campaign (SiSEC)](https://sisec.inria.fr/)
  * ****`CHALLENGE`**** [CHiME Speech Separation and Recognition Challenge](http://spandh.dcs.shef.ac.uk/chime_challenge/)

## Speaker Authentication
  * See [Speaker Verification](#speaker-verification)

## Speaker Diarisation
  * ****`WIKI`**** [Speaker diarisation](https://en.wikipedia.org/wiki/Speaker_diarisation)
  * ****`PAPER`**** [DNN-based speaker clustering for speaker diarisation](http://eprints.whiterose.ac.uk/109281/1/milner_is16.pdf)
  * ****`PAPER`**** [Unsupervised Methods for Speaker Diarization: An Integrated and Iterative Approach](http://groups.csail.mit.edu/sls/publications/2013/Shum_IEEE_Oct-2013.pdf)
  * ****`PAPER`**** [Audio-Visual Speaker Diarization Based on Spatiotemporal Bayesian Fusion](https://arxiv.org/pdf/1603.09725.pdf)
  * ****`CHALLENGE`**** [Rich Transcription Evaluation](https://www.nist.gov/itl/iad/mig/rich-transcription-evaluation)

## Speaker Recognition
  * ****`WIKI`**** [Speaker recognition](https://en.wikipedia.org/wiki/Speaker_recognition)
  * ****`PAPER`**** [A NOVEL SCHEME FOR SPEAKER RECOGNITION USING A PHONETICALLY-AWARE DEEP NEURAL NETWORK](https://pdfs.semanticscholar.org/204a/ff8e21791c0a4113a3f75d0e6424a003c321.pdf)
  * ****`PAPER`**** [DEEP NEURAL NETWORKS FOR SMALL FOOTPRINT TEXT-DEPENDENT SPEAKER VERIFICATION](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf)
  * ****`PAPER`**** [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/abs/1705.02304)
  * ****`PROJECT`**** [Voice Vector: which of the Hollywood stars is most similar to my voice?](https://github.com/andabi/voice-vector)
  * ****`CHALLENGE`**** [NIST Speaker Recognition Evaluation (SRE)](https://www.nist.gov/itl/iad/mig/speaker-recognition)
  * ****`INFO`**** [Are there any suggestions for free databases for speaker recognition?](https://www.researchgate.net/post/Are_there_any_suggestions_for_free_databases_for_speaker_recognition)
  * ****`DATA`**** [VoxCeleb2: Deep Speaker Recognition](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/)

## Speech Reading
  * See [Lip-reading](#lip-reading)

## Speech Recognition
  * See [Automatic Speech Recognition](#automatic-speech-recognition)

## Speech Segmentation
  * ****`WIKI`**** [Speech_segmentation](https://en.wikipedia.org/wiki/Speech_segmentation)
  * ****`PAPER`**** [Word Segmentation by 8-Month-Olds: When Speech Cues Count More Than Statistics](http://www.utm.toronto.edu/infant-child-centre/sites/files/infant-child-centre/public/shared/elizabeth-johnson/Johnson_Jusczyk.pdf)
  * ****`PAPER`**** [Unsupervised Word Segmentation and Lexicon Discovery Using Acoustic Word Embeddings](https://arxiv.org/abs/1603.02845)
  * ****`PAPER`**** [Unsupervised Lexicon Discovery from Acoustic Input](http://www.aclweb.org/old_anthology/Q/Q15/Q15-1028.pdf)
  * ****`PAPER`**** [Weakly supervised spoken term discovery using cross-lingual side information](http://www.research.ed.ac.uk/portal/files/29957958/1609.06530v1.pdf)
  * ****`DATA`**** [CALLHOME Spanish Speech](https://catalog.ldc.upenn.edu/ldc96s35)

## Speech Synthesis
  * ****`WIKI`**** [Speech synthesis](https://en.wikipedia.org/wiki/Speech_synthesis)
  * ****`PAPER`**** [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
  * ****`PAPER`**** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
  * ****`PAPER`**** [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
  * ****`PAPER`**** [Deep Voice 3: 2000-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1710.07654)
  * ****`PAPER`**** [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)  
  * ****`DATA`**** [The World English Bible](https://github.com/Kyubyong/tacotron)
  * ****`DATA`**** [LJ Speech Dataset](https://github.com/keithito/tacotron)
  * ****`DATA`**** [Lessac Data](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
  * ****`CHALLENGE`**** [Blizzard Challenge 2017](https://synsig.org/index.php/Blizzard_Challenge_2017)
  * ****`PRODUCT`**** [Lyrebird](https://lyrebird.ai/)
  * ****`PROJECT`**** [The Festvox project](http://www.festvox.org/index.html)
  * ****`TOOLKIT`**** [Merlin: The Neural Network (NN) based Speech Synthesis System](https://github.com/CSTR-Edinburgh/merlin)

## Speech Enhancement
  * ****`WIKI`**** [Speech enhancement](https://en.wikipedia.org/wiki/Speech_enhancement)
  * ****`BOOK`**** [Speech enhancement: theory and practice](https://www.amazon.com/Speech-Enhancement-Theory-Practice-Second/dp/1466504218/ref=sr_1_1?ie=UTF8&qid=1507874199&sr=8-1&keywords=Speech+enhancement%3A+theory+and+practice)
  * ****`PAPER`**** [An Experimental Study on Speech Enhancement BasedonDeepNeuralNetwork](http://staff.ustc.edu.cn/~jundu/Speech%20signal%20processing/publications/SPL2014_Xu.pdf)
  * ****`PAPER`**** [A Regression Approach to Speech Enhancement BasedonDeepNeuralNetworks](https://www.researchgate.net/profile/Yong_Xu63/publication/272436458_A_Regression_Approach_to_Speech_Enhancement_Based_on_Deep_Neural_Networks/links/57fdfdda08aeaf819a5bdd97.pdf)
  * ****`PAPER`**** [Speech Enhancement Based on Deep Denoising Autoencoder](https://www.researchgate.net/profile/Yu_Tsao/publication/283600839_Speech_enhancement_based_on_deep_denoising_Auto-Encoder/links/577b486108ae213761c9c7f8/Speech-enhancement-based-on-deep-denoising-Auto-Encoder.pdf)

## Speech-To-Text
  * See [Automatic Speech Recognition](#automatic-speech-recognition)

## Spoken Term Detection
  * See [Speech Segmentation](#speech-segmentation)

## Stemming
  * ****`WIKI`**** [Stemming](https://en.wikipedia.org/wiki/Stemming)
  * ****`PAPER`**** [A BACKPROPAGATION NEURAL NETWORK TO IMPROVE  ARABIC STEMMING](http://www.jatit.org/volumes/Vol82No3/7Vol82No3.pdf)
  * ****`TOOLKIT`**** [NLTK Stemmers](http://www.nltk.org/howto/stem.html)

## Term Extraction
  * ****`WIKI`**** [Terminology extraction](https://en.wikipedia.org/wiki/Terminology_extraction)
  * ****`PAPER`**** [Neural Attention Models for Sequence Classification: Analysis and Application to Key Term Extraction and Dialogue Act Detection](https://arxiv.org/pdf/1604.00077.pdf)

## Text Similarity
  * ****`WIKI`**** [Semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity)
  * ****`PAPER`**** [A Survey of Text Similarity Approaches](https://pdfs.semanticscholar.org/5b5c/a878c534aee3882a038ef9e82f46e102131b.pdf)
  * ****`PAPER`**** [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://casa.disi.unitn.it/~moschitt/since2013/2015_SIGIR_Severyn_LearningRankShort.pdf)
  * ****`PAPER`**** [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf)
  * ****`CHALLENGE`**** [SemEval-2014 Task 3: Cross-Level Semantic Similarity](http://alt.qcri.org/semeval2014/task3/)
  * ****`CHALLENGE`**** [SemEval-2014 Task 10: Multilingual Semantic Textual Similarity](http://alt.qcri.org/semeval2014/task10/)
  * ****`CHALLENGE`**** [SemEval-2017 Task 1: Semantic Textual Similarity](http://alt.qcri.org/semeval2017/task1/)
  * ****`WIKI`**** [Semantic Textual Similarity Wiki](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page)
  
## Text Simplification
  * ****`WIKI`**** [Text simplification](https://en.wikipedia.org/wiki/Text_simplification)
  * ****`PAPER`**** [Aligning Sentences from Standard Wikipedia to Simple Wikipedia](https://ssli.ee.washington.edu/~hannaneh/papers/simplification.pdf)
  * ****`PAPER`**** [Problems in Current Text Simplification Research: New Data Can Help](https://pdfs.semanticscholar.org/2b8d/a013966c0c5e020ebc842d49d8ed166c8783.pdf)
  * ****`DATA`**** [Newsela Data](https://newsela.com/data/)

## Text-To-Speech
  * See [Speech Synthesis](#speech-synthesis)

## Textual Entailment
  * ****`WIKI`**** [Textual entailment](https://en.wikipedia.org/wiki/Textual_entailment)
  * ****`PROJECT`**** [Textual Entailment with TensorFlow](https://github.com/Steven-Hewitt/Entailment-with-Tensorflow)
  * ****`PAPER`**** [Textual Entailment with Structured Attentions and Composition](https://arxiv.org/pdf/1701.01126.pdf)
  * ****`CHALLENGE`**** [SemEval-2014 Task 1: Evaluation of compositional distributional semantic models on full sentences through semantic relatedness and textual entailment](http://alt.qcri.org/semeval2014/task1/)
  * ****`CHALLENGE`**** [SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge](https://www.cs.york.ac.uk/semeval-2013/task7.html)

## Transliteration
  * ****`WIKI`**** [Transliteration](https://en.wikipedia.org/wiki/Transliteration)
  * ****`INFO`**** [Transliteration of Non-Latin scripts](http://transliteration.eki.ee/)
  * ****`PAPER`**** [A Deep Learning Approach to Machine Transliteration](https://pdfs.semanticscholar.org/54f1/23122b8dd1f1d3067cf348cfea1276914377.pdf)
  * ****`CHALLENGE`**** [NEWS 2016 Shared Task on Transliteration of Named Entities](http://workshop.colips.org/news2016/index.html)
  * ****`PROJECT`**** [Neural Japanese Transliteration—can you do better than SwiftKey™ Keyboard?](https://github.com/Kyubyong/neural_japanese_transliterator)
  
## Voice Conversion
  * ****`PAPER`**** [PHONETIC POSTERIORGRAMS FOR MANY-TO-ONE VOICE CONVERSION WITHOUT PARALLEL DATA TRAINING](http://www1.se.cuhk.edu.hk/~hccl/publications/pub/2016_paper_297.pdf)
  * ****`PROJECT`**** [Deep neural networks for voice conversion (voice style transfer) in Tensorflow](https://github.com/andabi/deep-voice-conversion)
  * ****`PROJECT`**** [An implementation of voice conversion system utilizing phonetic posteriorgrams](https://github.com/sesenosannko/ppg_vc)
  * ****`CHALLENGE`**** [Voice Conversion Challenge 2016](http://www.vc-challenge.org/vcc2016/index.html)
  * ****`CHALLENGE`**** [Voice Conversion Challenge 2018](http://www.vc-challenge.org/)
  * ****`DATA`**** [CMU_ARCTIC speech synthesis databases](http://festvox.org/cmu_arctic/)
  * ****`DATA`**** [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/ldc93s1)

## Voice Recognition
  * See [Speaker recognition](#speaker-recognition)

## Word Embeddings
  * ****`WIKI`**** [Word embedding](https://en.wikipedia.org/wiki/Word_embedding)
  * ****`TOOLKIT`**** [Gensim: word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
  * ****`TOOLKIT`**** [fastText](https://github.com/facebookresearch/fastText)
  * ****`TOOLKIT`**** [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
  * ****`INFO`**** [Where to get a pretrained model](https://github.com/3Top/word2vec-api)
  * ****`PROJECT`**** [Pre-trained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
  * ****`PROJECT`**** [Pre-trained word vectors of 30+ languages](https://github.com/Kyubyong/wordvectors)
  * ****`PROJECT`**** [Polyglot: Distributed word representations for multilingual NLP](https://sites.google.com/site/rmyeid/projects/polyglot)
  * ****`PROJECT`**** [BPEmb: a collection of pre-trained subword embeddings in 275 languages](https://github.com/bheinzerling/bpemb)
  * ****`CHALLENGE`**** [SemEval 2018 Task 10 Capturing Discriminative Attributes](https://competitions.codalab.org/competitions/17326)
  * ****`PAPER`**** [Bilingual Word Embeddings for Phrase-Based Machine Translation](https://ai.stanford.edu/~wzou/emnlp2013_ZouSocherCerManning.pdf)
  * ****`PAPER`**** [A Survey of Cross-Lingual Embedding Models](https://arxiv.org/abs/1706.04902)

## Word Prediction
  * ****`INFO`**** [What is Word Prediction?](http://www2.edc.org/ncip/library/wp/what_is.htm)
  * ****`PAPER`**** [The prediction of character based on recurrent neural network language model](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7960065)
  * ****`PAPER`**** [An Embedded Deep Learning based Word Prediction](https://arxiv.org/abs/1707.01662)
  * ****`PAPER`**** [Evaluating Word Prediction: Framing Keystroke Savings](http://aclweb.org/anthology/P08-2066)
  * ****`DATA`**** [An Embedded Deep Learning based Word Prediction](https://github.com/Meinwerk/WordPrediction/master.zip)
  * ****`PROJECT`**** [Word Prediction using Convolutional Neural Networks—can you do better than iPhone™ Keyboard?](https://github.com/Kyubyong/word_prediction)
  * ****`CHALLENGE`**** [SemEval-2018 Task 2, Multilingual Emoji Prediction](https://competitions.codalab.org/competitions/17344)

## Word Segmentation
  * ****`WIKI`**** [Word segmentation](https://en.wikipedia.org/wiki/Text_segmentation#Segmentation_problems)
  * ****`PAPER`**** [Neural Word Segmentation Learning for Chinese](https://arxiv.org/abs/1606.04300)
  * ****`PROJECT`**** [Convolutional neural network for Chinese word segmentation](https://github.com/chqiwang/convseg)
  * ****`TOOLKIT`**** [Stanford Word Segmenter](https://nlp.stanford.edu/software/segmenter.html)
  * ****`TOOLKIT`**** [NLTK Tokenizers](http://www.nltk.org/_modules/nltk/tokenize.html)

## Word Sense Disambiguation
  * ****`DATA`**** [Word-sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)
  * ****`PAPER`**** [Train-O-Matic: Large-Scale Supervised Word Sense Disambiguation in Multiple Languages without Manual Training Data](http://www.aclweb.org/anthology/D17-1008)
  * ****`DATA`**** [Train-O-Matic Data](http://trainomatic.org/data/train-o-matic-data.zip)
  * ****`DATA`**** [BabelNet](http://babelnet.org/)

