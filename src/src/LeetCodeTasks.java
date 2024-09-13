package src;

import java.util.Arrays;

public class LeetCodeTasks {

    // 1.2 Check Permutation: Given two strings, write a
    // method to decide if one is a permutation of the other.

    // 1 Solution: sort and check.
    // 2 Solution: calculate how many times each character
    // in first string and check the other string
    boolean permutation(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        int[] letters = new int[128];

        char[] sArray = s.toCharArray();
        for (char c : sArray) {
            letters[c]++;
        }
        for (int i = 0; i < t.length(); i++) {
            int c = t.charAt(i);
            letters[c]--;
            if (letters[c] < 0) {
                return false;
            }
        }
        return true;
    }


    // 303. Range Sum Query: Calculate the sum of
    // the elements of nums between indices left
    // and right inclusive where left <= right.

    // (prefix sum)
    // Calculate every sum in array, and pick by formula:
    // Sum(right) - Sum(left - 1)
    public static int sumRange(int[] nums, int left, int right) {
        for (int i = 1; i < nums.length; i++) {
            nums[i] += nums[i - 1];
        }

        if (left == 0) {
            return nums[right];
        }

        return nums[right] - nums[left - 1];
    }



}
