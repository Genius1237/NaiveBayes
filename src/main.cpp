#include <iostream>
#include <string>

using namespace std;

#include "naive_bayes.h"

int main() {
	int neg_max = 4, pos_min = 7;
	string train_bow_file = "../data/train.feat";
	string test_bow_file = "../data/test.feat";
	string vocab_file = "../data/imdb.vocab";
	string short_sw_file = "../data/sw_short";
	string long_sw_file = "../data/sw_long";

	cout << "Not removing stopwords\n";
	cout << "``````````````````````\n";
	NaiveBayes nb1(neg_max, pos_min, train_bow_file, vocab_file, "");
	cout << "Without binarization\n";
	nb1.test(test_bow_file, false);
	cout << "With binarization\n";
	nb1.test(test_bow_file, true);

	cout << "\nShort stopwords list\n";
	cout << "````````````````````\n";
	NaiveBayes nb2(neg_max, pos_min, train_bow_file, vocab_file, short_sw_file);
	cout << "Not using binarization\n";
	nb2.test(test_bow_file, false);
	cout << "Using binarization\n";
	nb2.test(test_bow_file, true);

	cout << "\nLong stopwords list\n";
	cout << "```````````````````\n";
	NaiveBayes nb3(neg_max, pos_min, train_bow_file, vocab_file, long_sw_file);
	cout << "Not using binarization\n";
	nb3.test(test_bow_file, false);
	cout << "Using binarization\n";
	nb3.test(test_bow_file, true);

	return 0;
}