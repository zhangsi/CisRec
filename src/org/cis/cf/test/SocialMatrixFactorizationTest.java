package org.cis.cf.test;


import org.cis.cf.algorithm.SocialMatrixFactorization;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.EpinionsRatingsReader;
import org.cis.io.EpinionsSparseBooleanMatrixReader;
import org.cis.matrix.SparseBooleanMatrix;

/**
 * This class tests the Social Matrix Factorization algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class SocialMatrixFactorizationTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.SocialMatrixFactorizationTes ../data/epinions/ratings_data_training.txt ../data/epinions/ratings_data_testing.txt 10 0.01 0.1 0.1 1 0.1 0.1 25 ../data/epinions/trust_data.txt 1 49290
		
		
		if(args.length != 13){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Number of latent factors");
			System.out.println("4, Learning rate");
			System.out.println("5, Regularization coeffient for user factors");
			System.out.println("6, Regularization coeffient for item factors");
			System.out.println("7, Biased learning rate");
			System.out.println("8, Regularization coeffient for user bias");
			System.out.println("9, Regularization coeffient for item bias");
			System.out.println("10, Max round of training");
			System.out.println("11, Social relations data path");
			System.out.println("12, Social regularization");
			System.out.println("13, user number");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			
			double learnRate = Double.parseDouble(args[3]);
			double userReg   = Double.parseDouble(args[4]);
			double itemReg   = Double.parseDouble(args[5]);
			
			double biasLearnRate = Double.parseDouble(args[6]);
			double biasUserReg   = Double.parseDouble(args[7]);
			double biasItemReg   = Double.parseDouble(args[8]);
			
			int maxIter = Integer.parseInt(args[9]);
			
			String socialFile = args[10];
			double socialReg  = Double.parseDouble(args[11]);
			
			int userNumber = Integer.parseInt(args[12]);
			
			EpinionsRatingsReader reader = new EpinionsRatingsReader();
			Ratings trainData  = reader.read(trainFile);
			Ratings testData   = reader.read(testFile);
			
			EpinionsSparseBooleanMatrixReader social_reader = new EpinionsSparseBooleanMatrixReader();
			SparseBooleanMatrix social_matrix = social_reader.read(socialFile, userNumber+1, userNumber+1);
			
			
			SocialMatrixFactorization recommender = new SocialMatrixFactorization(
					trainData,
					featureNumber,
					learnRate,
					userReg,
					itemReg,
					biasLearnRate,
					biasUserReg,
					biasItemReg,
					maxIter,
					social_matrix,
					socialReg
					);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}
