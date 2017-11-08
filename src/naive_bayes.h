#ifndef NAIVE_BAYES_H_
#define NAIVE_BAYES_H_

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <map>

#define ll long long
#define ld long double

using namespace std;

class NaiveBayes {
public:
	// Default constructor
	NaiveBayes();

	// Constructor
	NaiveBayes(
		int neg_max, int pos_min, const string& train_bow_file, 
		const string& vocab_file, const string& sw_file);
	
	// Prints stats after running on test data
	void test(const string& test_bow_file, bool use_bin);

private:
	// Returns a vector of words
	vector<string> readWords(const string& sw_file);

	// Returns true if the classification is correct
	bool singleTest(const string& bow_review, bool use_bin);

	// words_prob[0].first - without binarization 
	// words_prob[0].second - with binarization
	//                    .first - prob given positive
	//									  .second -  prob given negative
	vector<pair<pair<ld, ld>, pair<ld, ld>>> words_prob;

	vector<string> stop_words;
	vector<string> vocab_words;

	ll pos_reviews;
	ll neg_reviews;
	int neg_max;
	int pos_min;
	bool omit_sw;
};



#endif