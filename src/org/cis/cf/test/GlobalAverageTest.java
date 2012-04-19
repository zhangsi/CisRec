package org.cis.cf.test;

import org.cis.cf.algorithm.GlobalAverage;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

public class GlobalAverageTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.ProbabilisticMatrixFactorizationTest ../data/movielens/u1.base ../data/movielens/u1.test
		
		
		if(args.length != 2){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
		} else {
			String trainFile = args[0];
			String testFile = args[1];
			
			MovielensRatingsReader read = new MovielensRatingsReader();
			Ratings trainData  = read.read(trainFile);
			Ratings testData   = read.read(testFile);
			
			GlobalAverage recommender = new GlobalAverage(trainData);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}
