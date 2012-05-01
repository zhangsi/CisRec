package org.cis.io;

import org.cis.data.Ratings;

/**
 * This interface defines the function to read rating data from text file
 * 
 * @author Zhang Si
 *
 */
public interface RatingsReader {
	
	/**
	 * Read ratings data from text file
	 * 
	 * @param filePath
	 * @return Ratings of the data
	 */
	public Ratings read(String filePath);

}
