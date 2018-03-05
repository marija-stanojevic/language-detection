Project: Language detection on textual data , July 2015
Authors: Marija Stanojevic, Miroljub Enjakovic, Andrej Ciganj
Mentor: Momcilo Vasilijevic

Machine Learning Course, Summer 2015: Microsoft Development Center Serbia and Petnica Science Institute: http://psiml.petnica.rs/

This are materials from a two day project made during Machine Learning Course. It uses recurrent neural networks to detect language of provided text. 

Used tools: 
- Python (with PyCharm IDE) with theano and lassagne libraries
- Java with WikiXMLJ library for parsing data from wikipedia

Used texts can be found here: https://dumps.wikimedia.org/

Data from those dumps is parsed (using Java and WikiXMLJ library) into txt files which doesn't contain any interpunction characters, numbers or other charactes. Final parsed text was containing only words used in everyday conversation. Experiments were made with 3 (English, German, Italian) and 7 languages (English, German, Italian, Hungarian, Spanish, Portugese, Dutch).

Data was divided into training and validation sets, so that training set had 30000 words while validation set had 10000 words. Both sets were shuffled in order to prevent biases.

Tests were differing in:
- number of languages (3 or 7)
- network type used for training (RNN and LSTM)
- network architecture (1 and 2): picture of both can be found in project folder
- sequence length with which your network works in one moment (15, 25 and 30 characters in one moment)

In the code you can see comments which can help you to switch from training to test (and other way around) and to get results for different number of languages, network types and architectures.

Main file for starting the program is language_detection (1/2 depending on network architecture; inside each of them you can change other parameters).

Results can be found in Results folder. Each file name starts with type of network used, then it contains number of languages and sequence length and it has 'all' at the end if it uses network architecture 2.

Presentation, with results visualisation can be found here: https://sway.com/wJbKEG0zAlV32w3X
