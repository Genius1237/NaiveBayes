#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "naive_bayes.h"

using namespace std;

NaiveBayes::NaiveBayes() {
	;
}

NaiveBayes::NaiveBayes(
	int neg_max, int pos_min, const string& train_bow_file, 
	const string& vocab_file, const string& sw_file = "") {

	// set a few private variables
	pos_reviews = 0;
	neg_reviews = 0;
	this -> neg_max = neg_max;
	this -> pos_min = pos_min;
	omit_sw = false;
	if (sw_file != "") {
		stop_words = readWords(sw_file);
		omit_sw = true;
	}
	vocab_words = readWords(vocab_file);
	ll num_words = vocab_words.size();

	// setup words_freq and words_prob
	words_freq.resize(num_words);
	words_prob.resize(num_words);
	for (auto& word_info: words_freq) {
		word_info.first.first = 0;
		word_info.first.second = 0;
		word_info.second.first = 0;
		word_info.second.second = 0;
	}

	// populate words_freq
	ifstream in(train_bow_file);
	if (!in.is_open()) {
		cerr << "File opening failed\n";
		exit(0);
	}
	string line;
	ll vocab_size = 0;
	ll pos_wobin_freq = 0, neg_wobin_freq = 0, pos_wbin_freq = 0, neg_wbin_freq = 0; // total word frequencies
	
	// words_freq[0].first - without binarization 
	// words_freq[0].second - with binarization
	//                    .first - freq in positive reviews
	//									  .second -  freq in negative reviews
	vector<pair<pair<ll, ll>, pair<ll, ll>>> words_freq;

	// process each bow review in one iteration
	while (getline(in, line)) {

		// obtain sentiment of the review
		stringstream ss;
		ss.str(line);
		ll rating;
		bool is_pos;
		ss >> rating;
		if (rating <= neg_max) {
			is_pos = false;
			++neg_reviews;
		} else if (rating >= pos_min) {
			is_pos = true;
			++pos_reviews;
		} else {
			cerr << "Unexpected Neutral: " << rating << "\n";
			exit(0);
		}

		// process the words encoded as bow
		ll a, b;
		char discard;
		while (!ss.eof()) {
			ss >> a;
			ss.get(discard);
			ss >> b;
			ss.get(discard);
			if (omit_sw && binary_search(stop_words.begin(), stop_words.end(), vocab_words[a])) {
				continue;
			}
			if (is_pos) {
				words_freq[a].first.first += b;
				pos_wobin_freq += b;
				words_freq[a].second.first += 1;
				++pos_wbin_freq;
			} else {
				words_freq[a].first.second += b;
				neg_wobin_freq += b;
				words_freq[a].second.second += 1;
				++neg_wbin_freq;
			}
		}
	}

	in.close();

	// populate words_prob
	for (ll i = 0; i < num_words; i++) {
		words_prob[i].first.first = (1.0 + words_freq[i].first.first) / (pos_wobin_freq + num_words);
		words_prob[i].first.second = (1.0 + words_freq[i].first.second) / (neg_wobin_freq + num_words);
		words_prob[i].second.first = (1.0 + words_freq[i].second.first) / (pos_wbin_freq + num_words);
		words_prob[i].second.second = (1.0 + words_freq[i].second.second) / (neg_wbin_freq + num_words);
		//cout << words_prob[i].first.first << ","<< words_prob[i].first.second << endl;
	}
}

void NaiveBayes::test(const string& test_bow_file, bool use_bin) {
	ifstream in(test_bow_file);
	if (!in.is_open()) {
		cerr << "File opening failed\n";
		exit(0);
	}

	ll correct = 0, wrong = 0;
	string line;
	while (getline(in, line)) {
		if (singleTest(line, use_bin)) {
			++correct;
		} else {
			++wrong;
		}
	}

	in.close();
	cout << "Accuracy: " << fixed << setprecision(4) << (correct * 100.0) / (correct + wrong) << "\n";
}

vector<string> NaiveBayes::readWords(const string& sw_file) {
	ifstream fin(sw_file,ios::in);
	vector<string> data;

	while(!fin.eof()){
		string s;
		fin>>s;
		stringstream str(s);
		data.push_back(s);
	}
	return data;
}

bool NaiveBayes::singleTest(const string& bow_review, bool use_bin) {
	stringstream ss;
	ss.str(bow_review);
	ll rating;
	ss >> rating;
	bool is_pos;
	if (rating <= neg_max) {
		is_pos = false;
	} else if (rating >= pos_min) {
		is_pos = true;
	} else {
		cerr << "Unexpected Neutral: " << rating << "\n";
		exit(0);
	}

	ld pos_prob = log(static_cast<ld>(pos_reviews) / (pos_reviews + neg_reviews));
	ld neg_prob = log(static_cast<ld>(neg_reviews) / (pos_reviews + neg_reviews));
	ll a, b;
	char discard;
	while (!ss.eof()) {
		ss >> a;
		ss.get(discard);
		ss >> b;
		ss.get(discard);
		if (omit_sw && binary_search(stop_words.begin(), stop_words.end(), vocab_words[a])) {
			continue;
		}
		if (use_bin) {
			pos_prob += log(words_prob[a].second.first);
			neg_prob += log(words_prob[a].second.second);
		} else {
			pos_prob += b * log(words_prob[a].first.first);
			neg_prob += b * log(words_prob[a].first.second);
		}
	}
	return ((pos_prob >= neg_prob) == is_pos) ? true : false;
}