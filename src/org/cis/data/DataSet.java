package org.cis.data;

public interface DataSet {
	
	/**
	 * Build the user indices
	 */
	void BuildUserIndices();
	
	/**
	 * Build the item indices
	 */
	void BuildItemIndices();
	
	/**
	 * Build the random index
	 */
	void BuildRandomIndex();
	
	/**
	 * Get index for a given user and item
	 * @param user_id: the user ID
	 * @param item_id: the item ID
	 * @return: the index of the first event encountered that matches the user ID and item ID
	 */
	int GetIndex(int user_id, int item_id);
}
