package src;

import src.structures.ListNode;

import java.util.HashMap;
import java.util.Map;

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


    // 525. Contiguous Array
    // Longest array with same
    // amount of 1 and 0

    // Think like graph /\/\/\
    // If we have the same value of count for some place
    // Add this place to map
    // Get the max from map or from last counter
    public static int findMaxLength(int[] nums) {
        int n = nums.length;
        HashMap<Integer, Integer> map = new HashMap<>();
        int count = 0;
        int max = 0;
        map.put(0, 0);

        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) {
                count--;
            } else {
                count++;
            }

            if (map.containsKey(count)) {
                max = Math.max(max, i - map.get(count) + 1);
            } else {
                map.put(count, i + 1);
            }
        }

        return max;
    }

    // Reworked binary search with searching the start of shifting.
    public static int findLeft(int[] nums) {
        int n = nums.length;

        int left = 0;
        int right = n - 1;

        while (left < right) {
            int pivot = (left + right) / 2;
            if (nums[pivot] > nums[right]) {
                left = pivot + 1;
            } else {
                right = pivot;
            }
        }

        return left;
    }


    public static int searchInShifted(int[] nums, int target) {
        int left = findLeft(nums);
        int right = nums.length - 1;
        if (target >= nums[0] && left != 0) {
            right = left - 1;
            left = 0;
        }

        return binarySearch(nums, target);
    }

    // Binary search
    public static int binarySearch(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int pivot = left + (right - left) / 2;
            if (nums[pivot] == target) {
                return pivot;
            }
            if (nums[pivot] < target) {
                left = pivot + 1;
            } else {
                right = pivot - 1;
            }
        }

        return -1;
    }

    // Recursive binary search
    public static int binarySearchRec(int[] nums, int target, int left, int right) {
        // Base case: if left pointer exceeds right, the target is not found
        if (left > right) {
            return -1;
        }

        // Calculate mid to avoid overflow
        int mid = left + (right - left) / 2;

        // Check if the middle element is the target
        if (nums[mid] == target) {
            return mid;
        }

        // If the target is greater, search the right half
        if (nums[mid] < target) {
            return binarySearchRec(nums, target, mid + 1, right);
        }

        // If the target is smaller, search the left half
        return binarySearchRec(nums, target, left, mid - 1);
    }


    // 287. Find the Duplicate Number

    // If we have cycle fast and slow
    // will be on the same element
    // (in the middle of cycle).
    // After this we need to go to the
    // start of cycle to find duplicate link
    public static int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];

        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (fast != slow);

        int slow2 = nums[0];

        while (slow2 != slow) {
            slow = nums[slow];
            slow2 = nums[slow2];
        }

        return slow;
    }

    // 852. Peak Index in a Mountain Array
    //Input: arr = [0,10,5,2]
    //Output: 1 (return index of peak)

    // Binary search
    public static int peakIndexInMountainArray(int[] nums) {
        int left = 0;
        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if ((nums[mid] > nums[mid - 1]) && (nums[mid] > nums[mid + 1])) {
                return mid;
            }
            if (nums[mid] < nums[mid - 1]) {
                right = mid;
            }
            if (nums[mid] > nums[mid - 1]) {
                left = mid;
            }
        }

        return left;
    }

    // Two Sum 2: find indexes for two numbers in SORTED array
    // for constant space.
    // Input: nums = [2,7,11,15], target = 9
    // Output: [0,1]

    // Left and right pointers.
    // If sum less left++, if sum more right--;
    public static int[] twoSum2(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (nums[left] + nums[right] != target) {
            int sum = nums[left] + nums[right];
            if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[]{left, right};
    }


    // Two Sum2: find indexes for two numbers in unsorted array
    // Input: nums = [2,7,11,15], target = 9
    // Output: [0,1]

    // If we have needed number for sum, we return
    // Else we add this number to the map
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int needNum = target - nums[i];
            Integer mapValue = map.get(needNum);
            if (mapValue != null && mapValue != i) {
                return new int[]{i, map.get(needNum)};
            } else {
                map.put(nums[i], i);
            }
        }

        return null;
    }


    // 11. Container With Most Water
    //     |   | |                    |
    //     |   | |~~~~~~~~~~~~| |     |
    //     |   | |    | |     | |     |
    //     |---+-+----+-+-----+-+-----|
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;

        int maxContainer = 0;

        while (left < right) {
            int container = Math.min(height[left], height[right]) * (right - left);
            maxContainer = Math.max(container, maxContainer);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxContainer;
    }


    // 206. Reverse Linked List

    // current -> prev
    // prev = current
    // current = next
    public ListNode reverseList(ListNode head) {
        ListNode current = head;
        ListNode prev = null;

        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        return prev;
    }
}
