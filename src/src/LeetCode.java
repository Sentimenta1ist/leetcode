package src;

import org.junit.jupiter.api.Test;
import src.structures.ListNode;
import src.structures.TreeNode;
import src.structures.TreePrinter;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

public class LeetCode {


    private int[] arr(int... nums) {
        return nums;
    }

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


    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>(Comparator.reverseOrder());
        heap.add(Integer.MAX_VALUE);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                heap.add(matrix[i][j]);
                if (heap.size() >= k + 1)
                    heap.poll();

            }
        }
        return heap.peek();
    }

    public static class MedianFinder {
        // Max-heap to store the lower half of the numbers
        private PriorityQueue<Integer> lowerHalf;
        // Min-heap to store the upper half of the numbers
        private PriorityQueue<Integer> upperHalf;

        public MedianFinder() {
            // Max-heap (largest element on top)
            lowerHalf = new PriorityQueue<>(Collections.reverseOrder());
            // Min-heap (smallest element on top)
            upperHalf = new PriorityQueue<>();
        }

        public void addNum(int num) {
            // Add to lower half if it's empty or the number is less than or equal to the max of lowerHalf
            if (lowerHalf.isEmpty() || num <= lowerHalf.peek()) {
                lowerHalf.offer(num);
            } else {
                upperHalf.offer(num);
            }

            // Balance the heaps: if one heap has more than one element more than the other, rebalance
            if (lowerHalf.size() > upperHalf.size() + 1) {
                upperHalf.offer(lowerHalf.poll());
            } else if (upperHalf.size() > lowerHalf.size()) {
                lowerHalf.offer(upperHalf.poll());
            }
        }

        public double findMedian() {
            if (lowerHalf.size() == upperHalf.size()) {
                // If both heaps have the same size, return the average of the roots of both heaps
                return ((double) upperHalf.peek() + (double) lowerHalf.peek()) / 2.0;
            } else {
                // If heaps are not of the same size, the median is the root of the larger heap
                return lowerHalf.peek();
            }
        }

        public void removeNum(int num) {
            // Try to remove the number from the correct heap
            if (num <= lowerHalf.peek()) {
                lowerHalf.remove(num);  // O(n) time complexity
            } else {
                upperHalf.remove(num);  // O(n) time complexity
            }

            // Balance the heaps after removal
            balanceHeaps();
        }

        // Helper method to balance the heaps if needed
        private void balanceHeaps() {
            if (lowerHalf.size() > upperHalf.size() + 1) {
                upperHalf.offer(lowerHalf.poll());
            } else if (upperHalf.size() > lowerHalf.size()) {
                lowerHalf.offer(upperHalf.poll());
            }
        }
    }

    // 88. Merge Sorted Array
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;

        while (j >= 0) {
            if (i >= 0 && nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }

    // 27. Remove Element
    public static int removeElement(int[] nums, int val) {
        int counter = 0;
        int index = 0;
        int len = nums.length;
        while (index < len - counter) {
            if (nums[index] == val) {
                nums[index] = nums[len - counter - 1];
                counter++;
            } else {
                index++;
            }
        }
        return len - counter;
    }

    // 26. Remove Duplicates from Sorted Array
    public static int removeDuplicatesEasy(int[] nums) {
        int index = 1;

        for (int i = 1; i < nums.length; i++) {
            if (nums[index - 1] != nums[i]) {
                nums[index++] = nums[i];
            }
        }

        return index;
    }

    // 0 1 1 1 1 2 2 3 3 4

    // 80. Remove Duplicates from Sorted Array II
    public static int removeDuplicates(int[] nums) {
        int index = 1;
        int counter = 1;

        for (int i = 1; i < nums.length; i++) {
            if (nums[index - 1] != nums[i]) {
                nums[index++] = nums[i];
                counter = 1;
            } else if (nums[index - 1] == nums[i] && counter < 2) {
                nums[index++] = nums[i];
                counter++;
            }
        }

        return index;
    }

    // 169. Majority Element
    public static int majorityElement(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            Integer element = map.getOrDefault(num, 1);
            if (element > (nums.length / 2)) {
                return num;
            } else {
                map.put(num, element + 1);
            }
        }
        return nums[1];
    }

    //     189. Rotate Array
//     Given an integer array nums, rotate the array to the right by k steps, where k is non-negative
    public static void rotateArray(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    private static void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    public static ListNode partition(ListNode head, int x) {
        ListNode insertPlace = new ListNode(-101, head);
        while (insertPlace.next != null && insertPlace.next.val < x) {
            insertPlace = insertPlace.next;
        }

        ListNode pointer = insertPlace.next;
        while (pointer.next != null) {
            if (pointer.next.val < x) {
                ListNode tmp = new ListNode(pointer.next.val, insertPlace.next);
                insertPlace.next = tmp;
                insertPlace = insertPlace.next;
                pointer.next = pointer.next.next;
                if (insertPlace.next == head) {
                    head = tmp;
                }
            } else {
                pointer = pointer.next;
            }

        }
        return head;
    }

    public static boolean isSubPath(ListNode head, TreeNode root) {
        if (root == null) {
            return false; // If the tree node is null, we cannot have a match
        }
        // Check if the current node starts a matching path
        if (doesMatch(head, root)) {
            return true;
        }
        // Recursively check left and right subtrees
        return isSubPath(head, root.left) || isSubPath(head, root.right);
    }

    private static boolean doesMatch(ListNode head, TreeNode root) {
        if (head == null) {
            return true; // If we have reached the end of the list, it is a match
        }
        if (root == null) {
            return false; // If we reach a null tree node before the list ends, it's not a match
        }
        if (head.val != root.val) {
            return false; // If the current values do not match, it's not a match
        }
        // Continue checking both left and right subtrees
        return doesMatch(head.next, root.left) || doesMatch(head.next, root.right);
    }

    public static ListNode modifiedList(int[] nums, ListNode head) {
        HashSet<Integer> set = new HashSet<>(nums.length);
        for (int num : nums) {
            set.add(num);
        }

        while (head != null && set.contains(head.val)) {
            head = head.next;
        }

        ListNode pointer = head;

        while (pointer != null && pointer.next != null) {
            if (set.contains(head.val)) {
                pointer.next = pointer.next.next;
            } else {
                pointer = pointer.next;
            }
        }
        return head;
    }

    public static TreeNode insertNode(TreeNode node, int val) {
        if (node == null) {
            return new TreeNode(val);
        }
        if (node.val > val) {
            node.left = insertNode(node.left, val);
        } else if (node.val < val) {
            node.right = insertNode(node.right, val);
        }
        return node;
    }

    public static TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        int center = nums.length / 2;

        TreeNode root = insertNode(null, nums[center]);
        root.left = sortedArrayToBST(Arrays.copyOfRange(nums, 0, center));
        root.right = sortedArrayToBST(Arrays.copyOfRange(nums, center + 1, nums.length));

        return root;
    }

    public static TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null) {
            return root;
        }
        if (root.left == null) {
            root.left = root.right;
            root.right = null;
            root = invertTree(root.left);
        } else if (root.right == null) {
            root.right = root.left;
            root.left = null;
            root = invertTree(root.right);
        } else {
            TreeNode node = new TreeNode(root.left.val);
            root.left.val = root.right.val;
            root.right.val = node.val;
            root = invertTree(root.left);
            root = invertTree(root.right);
        }

        return root;
    }

    private static TreeNode reverseLevelsValues(TreeNode node, int level) {
        if (node == null) {
            return null;
        }
        if (node.left == null) {
            return node;
        }
        if (level % 2 != 0) {
            int val = node.left.val;
            node.left.val = node.right.val;
            node.right.val = val;
        }
        node.left = reverseLevelsValues(node.left, level + 1);
        node.right = reverseLevelsValues(node.right, level + 1);
        return node;
    }

    private static void reverseOddNodes(TreeNode node1, TreeNode node2, boolean isOddLevel) {
        if (node1 == null) {
            return;
        }

        if (isOddLevel) {
            int temp = node1.val;
            node1.val = node2.val;
            node2.val = temp;
        }
        reverseOddNodes(node1.left, node2.right, !isOddLevel);
        reverseOddNodes(node1.right, node2.left, !isOddLevel);
    }


    public static TreeNode reverseOddLevelsValues(TreeNode root) {
        reverseOddNodes(root.left, root.right, true);
        return root;

    }

    private static boolean isLessSubtree(TreeNode node, int rootValue) {
        if (node == null) {
            return true;
        }
        if (node.val >= rootValue) {
            return false;
        }
        return isLessSubtree(node.left, rootValue) && isLessSubtree(node.right, rootValue);
    }

    private static boolean isMoreSubtree(TreeNode node, int rootValue) {
        if (node == null) {
            return true;
        }
        if (node.val <= rootValue) {
            return false;
        }
        return isMoreSubtree(node.left, rootValue) && isMoreSubtree(node.right, rootValue);
    }


    public static boolean isValid(TreeNode root) {
        if (root == null) {
            return true;
        }

        return isLessSubtree(root.left, root.val) && isMoreSubtree(root.right, root.val);
    }

    static int kthSmallestCounter = 0;
    static int kthSmallestResult = 0;

    public static void helper(TreeNode root, int k) {
        if (root == null) {
            return;
        }

        helper(root.left, k);
        kthSmallestCounter++;
        if (k == kthSmallestCounter) {
            kthSmallestResult = root.val;
            return;
        }
        helper(root.right, k);
    }

    public static int kthSmallest(TreeNode root, int k) {
        helper(root, k);
        return kthSmallestResult;
    }

    public static void bfs(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            System.out.print(node.val + " ");
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
    }

    public static void traverseInOrder(TreeNode root) {
        if (root == null) {
            return;
        }
        traverseInOrder(root.left);
        System.out.print(root.val + " ");
        traverseInOrder(root.right);
    }

    // 530. Minimum Absolute Difference in BST

    // Given the root of a Binary Search Tree (BST),
    // return the minimum absolute difference between
    // the values of any two different nodes in the tree.

    static int minDiffInBst = Integer.MAX_VALUE;

    private static void traverseInOrder(TreeNode root, int prev) {
        if (root == null) {
            return;
        }

        traverseInOrder(root.left, root.val);

        if (Math.abs(root.val - prev) < minDiffInBst) {
            minDiffInBst = Math.abs(root.val - prev);
        }
        //System.out.println(root.val + ":" + prev);

        traverseInOrder(root.right, root.val);
    }

    public static int minDiffInBST(TreeNode root) {
        traverseInOrder(root, Integer.MAX_VALUE);

        return minDiffInBst;
    }


    // 442. Find All Duplicates in an Array

    // Solution is mark every element on current index with -
    // If now we have already negative element add to result
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> result = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            int index = Math.abs(nums[i]);
            if (nums[index - 1] < 0) {
                result.add(index);
            } else {
                nums[index - 1] *= -1;
            }
        }
        return result;
    }

    @Test
    public void testFindDuplicates() {
        assert Objects.equals(findDuplicates(new int[]{4, 3, 2, 7, 8, 2, 3, 1}), List.of(2, 3));
    }


    // 443. String Compression

    public int compress(char[] chars) {
        int slow = 0;
        int counter = 1;
        char current = chars[0];
        boolean inRepeat = false;
        for (int i = 1; i < chars.length; i++) {
            if (chars[i] != current) {
                chars[slow++ + 1] = chars[i];
                current = chars[i];
                counter = 1;
                inRepeat = false;
            } else {
                if (!inRepeat) {
                    inRepeat = true;
                    slow++;
                }
                if (chars[slow] == '9') {
                    chars[slow++] = '1';
                    chars[slow] = '0';
                    counter = 0;
                } else {
                    counter++;
                    chars[slow] = (char) (counter + '0');
                }
            }
        }

        return slow;
    }

    @Test
    public void testCompress() {
        char[] arr0 = new char[]{'a', 'b', 'c'};
        compress(arr0);
        System.out.println(arr0);
        char[] arr1 = new char[]{'a', 'a'};
        compress(arr1);
        System.out.println(arr1);
        char[] arr2 = new char[]{'a'};
        compress(arr2);
        System.out.println(arr2);
        char[] arr3 = new char[]{'a', 'a', 'b', 'b', 'b', 'c', 'c'};
        compress(arr3);
        System.out.println(arr3);
        char[] arr4 = new char[]{'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'};
        compress(arr4);
        System.out.println(arr4);

        int a = 3;
        char b = (char) a;
        System.out.println(b);
    }


    // 112. Path Sum

    // If we in the leaf (check target sum)
    // Else return false
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return targetSum - root.val == 0;
        } else {
            return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
        }
    }

    @Test
    public void testPathSum1() {
        TreeNode root = sortedArrayToBST(new int[]{1, 2, 3, 4, 5, 6, 7});
        TreePrinter.print(root);
        System.out.println(hasPathSum(root, 7));
    }


    public static void main(String[] args) {
        TreeNode root = sortedArrayToBST(new int[]{1, 2, 3, 4, 5, 6, 7});
        TreeNode root2 = sortedArrayToBST(new int[]{1, 2, 3, 4, 5, 6, 7});
        System.out.println(Objects.equals(root, root2));
        TreePrinter.print(root);
        //traverseInOrder(root);
        System.out.println(minDiffInBST(root));

    }

    // 121. Best Time to Buy and Sell Stock

    public int maxProfit(int[] arr) {
        int min = arr[0];
        int profit = 0;

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] - min > profit) {
                profit = arr[i] - min;
            }
            if (arr[i] < min) {
                min = arr[i];
            }
        }

        return profit;
    }

    public static void printMatrix(int[][] matrix) {
        System.out.println();
        for (int[] rows : matrix) {
            for (int element : rows) {
                System.out.printf("%4d", element);
            }
            System.out.println();
        }
    }

    private static void setOneZero(int[][] matrix, int row, int column) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i][column] = 0;
        }
        for (int j = 0; j < matrix.length; j++) {
            matrix[row][j] = 0;
        }
    }

    private static boolean isZeroPlus(int[][] matrix, int row, int column) {
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][column] != 0) {
                return false;
            }
        }
        for (int j = 0; j < matrix.length; j++) {
            if (matrix[row][j] != 0) {
                return false;
            }
        }
        return true;
    }

    // 73. Set Matrix Zeroes
    public static void setZeroes(int[][] matrix) {
        boolean fillColumn = false;
        boolean fillRow = false;

        for (int i = 0; i < matrix.length; i++)
            if (matrix[i][0] == 0)
                fillColumn = true;

        for (int j = 0; j < matrix[0].length; j++)
            if (matrix[0][j] == 0)
                fillRow = true;

        for (int i = 1; i < matrix.length; i++)
            for (int j = 1; j < matrix[0].length; j++)
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }

        for (int j = matrix[0].length - 1; j > 0; j--)
            if (matrix[0][j] == 0)
                for (int i = 0; i < matrix.length; i++)
                    matrix[i][j] = 0;

        for (int i = matrix.length - 1; i > 0; i--)
            if (matrix[i][0] == 0)
                for (int j = 1; j < matrix[0].length; j++)
                    matrix[i][j] = 0;

        if (fillColumn)
            for (int i = 0; i < matrix.length; i++)
                matrix[i][0] = 0;

        if (fillRow)
            for (int i = 0; i < matrix[0].length; i++)
                matrix[0][i] = 0;
    }


    // 153. Find Minimum in Rotated Sorted Array
    public static int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    public static int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n == 0) {
            return nums;
        }
        int[] result = new int[n - k + 1];
        LinkedList<Integer> dq = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (!dq.isEmpty() && dq.peek() < i - k + 1) {
                dq.poll();
            }
            while (!dq.isEmpty() && nums[i] >= nums[dq.peekLast()]) {
                dq.pollLast();
            }
            dq.offer(i);
            if (i - k + 1 >= 0) {
                result[i - k + 1] = nums[dq.peek()];
            }
        }
        return result;
    }

    public static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];

        int prefix = 1;
        for (int i = 0; i < n; i++) {
            res[i] = prefix;
            prefix = prefix * nums[i];
        }

        int post = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] = res[i] * post;
            post = post * nums[i];
        }

        return res;
    }

    private static int sumRange(int left, int right, int[] nums) {
        if (left != 0) {
            left = nums[left - 1];
        }
        return nums[right] - left;
    }

    public static int subarraySum(int[] nums, int k) {


        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);

        int sum = 0;
        int counter = 0;
        for (int num : nums) {
            sum = sum + num;
            if (map.containsKey(sum - k))
                counter += map.get(sum - k);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }

        return counter;
    }

    public static boolean isPalindrome(ListNode head) {
        ListNode prev = null;
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next;
            fast = fast.next;
            ListNode tmp = slow.next;
            slow.next = prev;
            prev = slow;
            slow = tmp;

        }

        if (fast != null) slow = slow.next;


        while (slow != null && prev != null) {
            if (slow.val != prev.val) return false;
            slow = slow.next;
            prev = prev.next;
        }

        return true;
    }

    @Test
    public void testPalindromeList() {
        ListNode node = new ListNode(1, new ListNode(3, new ListNode(3, new ListNode(2, new ListNode(1, null)))));
        ListNode.print(node);
        System.out.println(isPalindrome(node));
        System.out.println("------------");
        ListNode node2 = new ListNode(1, new ListNode(2, new ListNode(2, new ListNode(1, null))));
        ListNode.print(node2);
        System.out.println(isPalindrome(node2));
        System.out.println("------------");
        ListNode node3 = new ListNode(1, null);
        ListNode.print(node3);
        System.out.println(isPalindrome(node3));
        System.out.println("------------");
        //ListNode.print(node);
    }


    public static double[] medianSlidingWindow(int[] nums, int k) {

        int n = nums.length;
        double[] res = new double[n - k + 1];

        MedianFinder finder = new MedianFinder();

        for (int i = 0; i < k; i++) {
            finder.addNum(nums[i]);
        }

        res[0] = finder.findMedian();

        for (int i = k; i < n; i++) {
            finder.removeNum(nums[i - k]);
            finder.addNum(nums[i]);
            res[i - k + 1] = finder.findMedian();
        }

        System.out.println(finder.findMedian());

        return res;
    }

    public static ListNode removeElements(ListNode head, int val) {

        ListNode current = head;

        while (current != null && current.val == val) {
            head = current.next;
            current = head;
        }

        ListNode prev = new ListNode(0, head);

        while (current != null) {
            if (current.val == val) {
                prev.next = current.next;
            } else {
                prev = current;
            }
            current = prev.next;
        }

        return head;
    }

    public static int rob(int[] nums) {
        int n = nums.length;
        int[] maxRob = new int[n];
        if (n == 1) {
            return nums[0];
        }
        maxRob[0] = nums[0];
        maxRob[1] = Math.max(nums[0], nums[1]);
        int max = maxRob[1];
        for (int i = 2; i < n; i++) {
            maxRob[i] = Math.max((maxRob[i - 1] - nums[i - 1]) + nums[i], maxRob[i - 2] + nums[i]);
            if (maxRob[i] > max) {
                max = maxRob[i];
            }
        }

        return max;
    }

    public static int maxSubArray(int[] nums) {
        int n = nums.length;
        int maxSum = nums[0];
        int currSum = 0;
        for (int num : nums) {
            if (currSum < 0) {
                currSum = 0;
            }
            currSum += num;
            if (currSum > maxSum) {
                maxSum = currSum;
            }
        }

        return maxSum;
    }

    static int[] memoCoinChange;

    private static int helperCoinChange(int[] coins, int amount, int level) {
        int res = Integer.MAX_VALUE;
        if (amount == 0) {
            return 0;
        }
        if (amount < 0) {
            return Integer.MAX_VALUE;
        }
        if (memoCoinChange[amount] != -1) {
            return memoCoinChange[amount];
        }
        for (int i = coins.length - 1; i >= 0; i--) {
            int currentResult = helperCoinChange(coins, amount - coins[i], level);
            if (currentResult != Integer.MAX_VALUE) {
                res = Math.min(res, currentResult + 1);
            }
        }
        memoCoinChange[amount] = res;
        return res;
    }

    public static int coinChange(int[] coins, int amount) {
        memoCoinChange = new int[amount + 1];
        Arrays.fill(memoCoinChange, -1);
        int answer = helperCoinChange(coins, amount, 0);
        return answer == Integer.MAX_VALUE ? -1 : answer;
    }


    private static void backtrack(TreeNode root, StringBuilder sb, List<String> res) {
        int length = sb.length();

        if (root.left == null && root.right == null) {
            res.add(sb.toString());
        }
        if (root.right != null) {
            backtrack(root.right, sb.append("->").append(root.right.val), res);
            sb.setLength(length);
        }
        if (root.left != null) {
            backtrack(root.left, sb.append("->").append(root.left.val), res);
            sb.setLength(length);
        }
    }

    public static List<String> binaryTreePaths(TreeNode root) {
        StringBuilder sb = new StringBuilder(root.val + "");
        List<String> res = new ArrayList<>();
        backtrack(root, sb, res);
        return res;
    }

    static char[][] boardWordExist;
    static String wordExist;


    private static boolean helper(int i, int j, int index) {
        if (index == wordExist.length())
            return true;
        if (i < 0 || i > boardWordExist.length - 1 || j < 0 || j > boardWordExist[0].length - 1 || boardWordExist[i][j] != wordExist.charAt(index))
            return false;

        char a = boardWordExist[i][j];
        boardWordExist[i][j] = '_';
        boolean res = helper(i - 1, j, index + 1) || helper(i + 1, j, index + 1) || helper(i, j - 1, index + 1) || helper(i, j + 1, index + 1);
        boardWordExist[i][j] = a;
        return res;
    }

    public static boolean exist(char[][] board1, String word1) {

        boardWordExist = board1;
        wordExist = word1;
        for (int i = 0; i < boardWordExist.length; i++) {
            for (int j = 0; j < boardWordExist[0].length; j++) {
                if (boardWordExist[i][j] == wordExist.charAt(0)) {
                    if (helper(i, j, 0)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }


    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(Collections.emptyList());
        for (int num : nums) {
            int size = res.size();
            for (int j = 0; j < size; j++) {
                List<Integer> oldSubset = res.get(j);
                List<Integer> subset = new ArrayList<>(oldSubset);
                subset.add(num);
                res.add(subset);
            }
        }

        return res;
    }


    public List<List<Integer>> helper(List<List<Integer>> permut, int n) {
        List<List<Integer>> res = new ArrayList<>();
        for (List<Integer> integers : permut) {
            for (int j = 0; j < integers.size() + 1; j++) {
                List<Integer> permutation = new ArrayList<>(integers);
                permutation.add(j, n);
                res.add(permutation);
            }
        }
        return res;
    }


    public List<List<Integer>> permute2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> first = new ArrayList<>();
        first.add(nums[0]);
        res.add(first);

        for (int i = 1; i < nums.length; i++) {
            res = helper(res, nums[i]);
        }

        return res;
    }

    List<List<Integer>> result = new ArrayList<>();

    private List<Integer> createListFromArray(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        for (int num : nums) {
            ans.add(num);
        }
        return ans;
    }

    private void swap(int i1, int i2, int[] nums) {
        int tmp = nums[i1];
        nums[i1] = nums[i2];
        nums[i2] = tmp;
    }

    private void backtrack(int current, int[] nums) {
        if (current == nums.length) {
            result.add(createListFromArray(nums));
        }

        for (int i = current; i < nums.length; i++) {
            swap(current, i, nums);
            backtrack(current + 1, nums);
            swap(current, i, nums);
        }
    }

    public List<List<Integer>> permute(int[] nums) {

        backtrack(0, nums);
        return result;
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums)
            set.add(num);

        int max = 0;
        for (int num : nums) {
            int count = 1;
            int right = num;
            while (set.contains(++right)) {
                count++;
                set.remove(right);
            }
            int left = num;
            while (set.contains(++left)) {
                count++;
                set.remove(left);
            }
            max = Math.max(count, max);
        }
        return max;
    }

    Map<String, Boolean> map = new HashMap<>();

    private String getSubs(String s, int j) {
        if (j > s.length()) {
            return s;
        } else {
            return s.substring(0, j);
        }
    }

    Map<String, Boolean> memoWordBreak = new HashMap<>();

    private boolean helper(int start, String s, List<String> wordDict) {
        String str = s.substring(start); // Получаем подстроку от позиции start

        if (str.isEmpty()) return true; // Базовый случай - если подстрока пуста
        if (memoWordBreak.containsKey(str)) return memoWordBreak.get(str); // Проверяем, есть ли результат в кэше

        for (String w1 : wordDict) {
            // Если длина слова меньше или равна оставшейся подстроке и она совпадает с началом подстроки
            if (w1.length() <= str.length() && w1.equals(str.substring(0, w1.length()))) {
                // Рекурсивно проверяем остаток строки
                if (helper(start + w1.length(), s, wordDict)) {
                    memoWordBreak.put(str, true); // Сохраняем результат для подстроки
                    return true;
                }
            }
        }
        memoWordBreak.put(str, false); // Сохраняем результат если не удалось разбить строку
        return false;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        return helper(0, s, wordDict); // Начинаем с позиции 0
    }

    public boolean wordBreak2(String s, List<String> wordDict) {

        //wordDict.sort(Comparator.comparing(s.length()));

//        Set<String> set = new HashSet<>(wordDict);
//
//        int startIndex = 0;
//        for (int i = 0; i <= s.length(); i++) {
//            if (set.contains(s.substring(startIndex, i))) {
//                startIndex = i;
//            }
//            if (set.contains(s.substring(startIndex))) {
//                startIndex = s.length();
//                break;
//            }
//        }
//
//        return startIndex == s.length();
        return false;
    }

    private void sinkTheIsland(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length) {
            return;
        }
        if (j < 0 || j >= grid[0].length) {
            return;
        }
        if (grid[i][j] == 0) {
            return;
        }
        grid[i][j] = 0;
        sinkTheIsland(grid, i + 1, j);
        sinkTheIsland(grid, i - 1, j);
        sinkTheIsland(grid, i, j + 1);
        sinkTheIsland(grid, i, j - 1);
    }

    public int numIslands(char[][] grid) {
        int counter = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    sinkTheIsland(grid, i, j);
                    counter++;
                }
            }
        }
        return counter;
    }


    static class Rec {

        boolean pac;
        boolean atl;

        Rec(boolean a, boolean b) {
            pac = a;
            atl = b;
        }

    }

    Rec[][] memo1;

    private Rec waterfall(int[][] heights, int i, int j, int prev) {
        if (i < 0 || j < 0) {
            return new Rec(true, false);
        }
        if (i >= heights.length || j >= heights[0].length) {
            return new Rec(false, true);
        }
        if (heights[i][j] > prev) {
            return new Rec(false, false);
        }
        if (memo1[i][j] != null && memo1[i][j].atl && memo1[i][j].pac) {
            return new Rec(true, true);
        }

        int previous = heights[i][j];
        heights[i][j] = Integer.MAX_VALUE;
        Rec s1 = waterfall(heights, i + 1, j, previous);
        Rec s2 = waterfall(heights, i - 1, j, previous);
        Rec s3 = waterfall(heights, i, j + 1, previous);
        Rec s4 = waterfall(heights, i, j - 1, previous);
        boolean pac = s1.pac || s2.pac || s3.pac || s4.pac;
        boolean atl = s1.atl || s2.atl || s3.atl || s4.atl;
        memo1[i][j] = new Rec(pac, atl);
        heights[i][j] = previous;

        return memo1[i][j];
    }

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> res = new ArrayList<>();
        memo1 = new Rec[heights.length][heights[0].length];

        for (int i = 0; i < heights.length; i++) {
            for (int j = 0; j < heights[0].length; j++) {
                Rec r1 = waterfall(heights, i, j, Integer.MAX_VALUE);
                if (r1.pac && r1.atl) {
                    res.add(List.of(i, j));
                }
            }
        }

        return res;
    }

    @Test
    public void testPacificAtlantic() {
        int[][] arr = new int[][]{
                {1, 2, 2, 3, 5},
                {3, 3, 3, 4, 4},
                {2, 4, 5, 3, 1},
                {6, 7, 1, 4, 5},
                {5, 1, 1, 2, 4}};

        int[][] arr2 = new int[][]{
                {7, 7, 7, 1, 7},
                {7, 6, 6, 2, 7},
                {7, 6, 6, 3, 7},
                {7, 6, 6, 6, 7},
                {7, 6, 6, 5, 6}};

        int[][] arr3 = new int[][]{
                {9, 9, 9},
                {9, 1, 9},
                {9, 9, 9},};

        System.out.println(pacificAtlantic(arr3));
    }


    public void backtrack(char[] str, int pos, List<String> res) {
        if (pos == str.length - 1) {
            res.add(new String(str));
            return;
        }
        char element = str[pos];
        if (Character.isAlphabetic(element)) {
            if (Character.isUpperCase(element)) {
                str[pos] = Character.toLowerCase(element);
                backtrack(str, pos + 1, res);
                str[pos] = Character.toUpperCase(element);

            } else {
                str[pos] = Character.toUpperCase(element);
                backtrack(str, pos + 1, res);
                str[pos] = Character.toLowerCase(element);
            }
        }
        backtrack(str, pos + 1, res);
    }

    public List<String> letterCasePermutation(String s) {
        List<String> res = new ArrayList<>();
        backtrack(s.toCharArray(), 0, res);
        return new ArrayList<>(res);
    }


    public void backtrack(int[] nums, boolean[] used, List<Integer> permutation, HashSet<List<Integer>> res) {
        if (permutation.size() == nums.length && !res.contains(permutation)) {
            res.add(new ArrayList<>(permutation));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                permutation.add(nums[i]);
                used[i] = true;
                backtrack(nums, used, permutation, res);
                permutation.removeLast();
                used[i] = false;
            }
        }
    }


    public void comb(int[] nums, int k, List<Integer> combination, List<List<Integer>> res) {
        if (combination.size() >= k) {
            res.add(new ArrayList<>(combination));
        } else {
            for (int i = 0; i < nums.length; i++) {
                combination.add(nums[i]);
                comb(Arrays.copyOfRange(nums, i + 1, nums.length), k, combination, res);
                combination.removeLast();
            }
        }
    }

    public void comb2(int[] nums, int start, int k, List<Integer> combination, List<List<Integer>> res) {
        if (combination.size() >= k) {
            res.add(new ArrayList<>(combination));
        } else {
            for (int i = start; i < nums.length; i++) {
                combination.add(nums[i]);
                comb2(nums, i + 1, k, combination, res);
                combination.removeLast();
            }
        }
    }

    public List<List<Integer>> combinations(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();

        comb2(new int[]{1, 2, 3, 4}, 0, k, new ArrayList<>(), res);
        return new ArrayList<>(res);
    }


    private void backtrack(int[] candidates, int start, int target, List<Integer> combinations, List<List<Integer>> res) {
        if (target < 0) {
            return;
        }

        if (0 == target) {
            res.add(new ArrayList<>(combinations));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            combinations.add(candidates[i]);
            backtrack(candidates, i, target - candidates[i], combinations, res);
            combinations.removeLast();
        }
    }


    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(candidates, 0, target, new ArrayList<>(), res);
        return res;
    }

    int helper(int[] nums, int l, int r, int res) {
        if (res != -1) {
            return res - 1;
        }
        int res1 = -1;
        int pivot = (l + r) / 2;
        if (l == pivot || r == pivot) {
            return res1;
        }
        if (nums[pivot] > nums[pivot - 1] && nums[pivot] > nums[pivot + 1]) {
            res1 = pivot;
        }


        return Math.max(helper(nums, pivot, r, res1), helper(nums, l, pivot, res1));
    }

    public int findPeakElement(int[] nums) {
        Arrays.sort(nums);
        int[] all = new int[nums.length + 2];
        all[0] = Integer.MIN_VALUE;
        System.arraycopy(nums, 0, all, 1, all.length - 1 - 1);
        all[all.length - 1] = Integer.MIN_VALUE;
        System.out.println(Arrays.toString(all));
        int result = helper(all, 0, all.length - 1, -1);

        return result == -1 ? 0 : result;
    }

    public int minSubArrayLen(int target, int[] a) {

        int left = 0;
        int right = 1;
        int sum = a[left];
        int minsSize = Integer.MAX_VALUE;

        while (left < right) {
            while (sum >= target) {
                minsSize = Math.min(right - left, minsSize);
                sum -= a[left];
                left++;
            }

            if (right < a.length) {
                sum += a[right++];
            } else {
                left++;
            }

        }

        return minsSize == Integer.MAX_VALUE ? 0 : minsSize;
    }

    private int uniqueNumber(int[] nums, int start, int end) {
        HashSet<Integer> set = new HashSet<>();
        for (int i = start; i < end; i++) {
            set.add(nums[i]);
        }
        return set.size();
    }

    private int getMinIndex(Map<Integer, Integer> map) {
        int minIndex = Integer.MAX_VALUE;
        for (Integer value : map.values()) {
            minIndex = Math.min(value, minIndex);
        }
        return minIndex;
    }

    public int totalFruit(int[] fruits) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int maxCount = 0, left = 0, right = 0;

        while (right < fruits.length) {
            map.put(fruits[right], right++);
            if (map.size() > 2) {
                int minIndex = getMinIndex(map);
                map.remove(fruits[minIndex]);
                left = minIndex + 1;
            }
            maxCount = Math.max(right - left, maxCount);
        }
        return maxCount;
    }

    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            map.put(s1.charAt(i), map.getOrDefault(s1.charAt(i), 0) + 1);
        }

        char[] str = s2.toCharArray();

        int l = 0, r = 0, count = 0;
        while (r < str.length) {
            char element = str[r];
            int amount = map.getOrDefault(element, 0);
            r++;
            if (amount == 0) {
                while (str[l++] != element) {
                    map.put(str[l - 1], map.getOrDefault(str[l - 1], 0) + 1);
                }
            } else {
                map.put(element, amount - 1);
            }
            if (r - l >= s1.length()) {
                return true;
            }
        }

        return false;
    }


    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        Set<Character> set = new HashSet<>();

        int left = 0, right = 0;
        char[] arr = s.toCharArray();
        int maxLen = 0;
        while (right < arr.length) {
            if (set.contains(arr[right])) {
                while (arr[left++] != arr[right]) {
                    set.remove(arr[left - 1]);
                }
            }
            set.add(arr[right++]);
            maxLen = Math.max(right - left, maxLen);
        }
        return maxLen;
    }


    public int characterReplacement(String s, int k) {
        Map<Character, Integer> map = new HashMap<>();
        s.isEmpty();
        int left = 0, right = 0, allSum = 0, sumMax = 0;
        char[] arr = s.toCharArray();
        int res = 0;
        while (right < arr.length) {
            int currFreq = map.getOrDefault(arr[right], 0) + 1;
            map.put(arr[right], currFreq);
            allSum += 1;
            sumMax = Math.max(currFreq, sumMax);
            right++;
            while (allSum - sumMax > k) {
                int leftFreq = map.get(arr[left]);
                map.put(arr[left], leftFreq - 1);
                if (leftFreq == sumMax) {
                    sumMax = map.get(arr[left + 1]);
                }
                allSum -= 1;
                left++;
            }

            res = Math.max(res, right - left);
        }
        return sumMax;
    }

    class Point {
        int x;
        int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static double squaredDistance(Point p1, Point p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
    }

    public int[][] kClosest(int[][] points, int k) {
        Point reference = new Point(0, 0);


        // Custom comparator based on squared distance to the reference point
        Comparator<Point> pointComparator = (p1, p2) -> {
            double dist1 = squaredDistance(p1, reference);
            double dist2 = squaredDistance(p2, reference);
            return Double.compare(dist1, dist2); // Compare based on distance
        };

        PriorityQueue<Point> pq = new PriorityQueue<>(pointComparator);

        for (int i = 0; i < points.length; i++) {
            pq.add(new Point(points[i][0], points[i][1]));
        }

        int[][] res = new int[k][2];
        for (int i = 0; i < k; i++) {
            Point p = pq.poll();
            res[i][0] = p.x;
            res[i][1] = p.y;
        }
        return res;
    }

    @Test
    public void testKClosest() {
        int[][] res = kClosest(new int[][]{{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}}, 2);
        printMatrix(res);
    }

    private void backtrackSubsets2(int[] nums, int start, List<Integer> subs, List<List<Integer>> res) {
        if (!res.contains(subs)) {
            res.add(new ArrayList<>(subs));
        }
        for (int i = start; i < nums.length; i++) {
            subs.add(nums[i]);
            backtrackSubsets2(nums, i + 1, subs, res);
            subs.removeLast();

        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        backtrackSubsets2(nums, 0, new ArrayList<>(), res);
        return res;
    }

    @Test
    public void testSubsets() {
        System.out.println(subsetsWithDup(new int[]{1, 2, 2}));
        System.out.println(subsetsWithDup(new int[]{4, 4, 4, 1, 4}));
    }

    private int checkPalindrome(int left, int right, char[] s) {
        while (left >= 0 && right < s.length && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    public String longestPalindrome(String s) {
        int maxLen = 0, start = -1;
        char[] str = s.toCharArray();
        for (int i = 0; i < str.length; i++) {
            int len = Math.max(checkPalindrome(i, i + 1, str), checkPalindrome(i, i, str));
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }
        return s.substring(start, start + maxLen);
    }

    private void backtrackSum2(int[] candidates, int target, int start, List<Integer> comb, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(comb));
            return;
        }
        if (target < 0) {
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            comb.add(candidates[i]);
            backtrackSum2(candidates, target - candidates[i], i + 1, comb, res);
            comb.removeLast();
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrackSum2(candidates, target, 0, new ArrayList<>(), res);
        return res;
    }

    private void backtrack(int k, int n, int start, List<Integer> comb, List<List<Integer>> res) {
        if (n < 0) return;
        if (comb.size() > k) {
            return;
        }
        if (n == 0 && comb.size() == k) {
            res.add(new ArrayList<>(comb));
            return;
        }
        for (int i = start; i < 10; i++) {
            comb.add(i);
            backtrack(k, n - i, i + 1, comb, res);
            comb.removeLast();
        }
    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(k, n, 1, new ArrayList<>(), res);
        return res;
    }

    @Test
    public void testCombSum3() {
        System.out.println(combinationSum3(3, 7));
        System.out.println(combinationSum3(3, 9));
    }


    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        int level = 0;
        queue.add(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size(); // Number of nodes at the current level

            // Process all nodes for the current level
            double sum = 0;
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.remove();
                sum += node.val;
                System.out.println(node.val + " " + "level: " + level);

                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            result.add((double) sum / levelSize);

            level++; // Increment level after processing the current level
        }

        return result;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return null;

        List<List<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.remove();
                //level.add(node.val);
                level.add(0, node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            result.add(level);
        }
        return result;
    }


    @Test
    public void testAverageOfLevels() {
        // Manually creating the tree from the BFS order
        TreeNode root = new TreeNode(1); // Level 0

        // Level 1
        root.left = new TreeNode(4);
        root.right = new TreeNode(20);

        // Level 2
        root.left.left = new TreeNode(1);
        ; // Null as per the input array
        root.left.right = new TreeNode(3);
        root.right.left = new TreeNode(8);
        root.right.right = new TreeNode(10); // Null as per the input array

        // Level 3
        root.left.right.left = new TreeNode(12);
        root.left.right.right = null; // Null as per the input array
        root.right.left.left = new TreeNode(7);
        root.right.left.right = new TreeNode(8);

        TreePrinter.print(root);

        System.out.println(levelOrder(root));
    }


    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingDouble(o -> o[0]));
        int[][] result = new int[intervals.length][2];
        result[0] = new int[]{intervals[0][0], intervals[0][1]};

        int counter = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] <= result[counter][1] && intervals[i][1] > result[counter][1]) {
                result[counter][1] = intervals[i][1];
            } else if (intervals[i][1] > result[counter][1]) {
                result[++counter][0] = intervals[i][0];
                result[counter][1] = intervals[i][1];
            }
//            else {
//                result[++counter][0] = intervals[i][0];
//                result[counter][1] = intervals[i][1];
//            }
        }

        int[][] newArray = new int[counter + 1][2];
        for (int i = 0; i <= counter; i++) {
            newArray[i][0] = result[i][0];
            newArray[i][1] = result[i][1];
        }

        return newArray;
    }

    @Test
    public void testMergeIntervals() {
        // merge(new int[][]{{2, 6}, {8, 10}, {15, 18}, {1, 3}});

        printMatrix(merge(new int[][]{{15, 18}, {16, 17}, {1, 4}, {2, 7}}));
        printMatrix(merge(new int[][]{{1, 3}, {2, 6}, {2, 4}}));

        System.out.println(1 ^ 1);
    }

    public boolean canPartition(int[] nums) {
        Arrays.sort(nums);

        int left = 0, right = nums.length - 1;

        int sum1 = nums[left];
        int sum2 = nums[right];
        while (left + 1 < right) {
            if (sum1 > sum2) {
                right--;
                sum2 += nums[right];
            } else {
                left++;
                sum1 += nums[left];
            }
        }


        return sum1 == sum2;
    }

    @Test
    public void canPartition() {
        System.out.println(canPartition(new int[]{1, 5, 11, 5}));
    }


    int size = 0;

    public void dfs(int[][] grid, int i, int j, int depth) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == 0) {
            return;
        }

        grid[i][j] = 0;
        size++;
        dfs(grid, i + 1, j, depth + 1);
        dfs(grid, i, j + 1, depth + 1);
        dfs(grid, i - 1, j, depth + 1);
        dfs(grid, i, j - 1, depth + 1);
    }

    public int maxAreaOfIsland(int[][] grid) {
        int maxArea = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    size = 0;
                    dfs(grid, i, j, 0);
                    maxArea = Math.max(maxArea, size);
                }
            }
        }

        return maxArea;
    }

    @Test
    public void maxAreaOfIslands() {
        int[][] grid = {
                {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0},
                {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}
        };

        int[][] grid2 = {
                {0, 0, 0},
                {1, 1, 1},
                {0, 1, 0}
        };

        System.out.println(maxAreaOfIsland(grid2));
    }

    private int res = 0;

    private void backtrack(int[] nums, int target, int curr, int start) {
        if (curr == target && start == nums.length) {
            res++;
            return;
        }
        if (start == nums.length) {
            return;
        }
        backtrack(nums, target, curr + nums[start], start + 1);
        backtrack(nums, target, curr - nums[start], start + 1);
    }

    public int findTargetSumWays(int[] nums, int target) {
        backtrack(nums, target, 0, 0);
        return res;
    }

    @Test
    public void testFindTargetSum() {
        res = 0;
        System.out.println(findTargetSumWays(new int[]{1, 2, 2, 3}, 0));
        res = 0;
        System.out.println(findTargetSumWays(new int[]{1, 0}, 1));
        res = 0;
        System.out.println(findTargetSumWays(new int[]{1, 1, 1, 1, 1}, 3));
        res = 0;
        System.out.println(findTargetSumWays(new int[]{1}, 1));
    }

    int max = Integer.MIN_VALUE;

    public void maxProductHelper(int[] nums, int start) {
        if (start >= nums.length) {
            return;
        }
        int product = 1;
        for (int i = start; i < nums.length; i++) {
            product *= nums[i];
            max = Math.max(max, product);
        }
        maxProductHelper(nums, start + 1);
    }

    public int maxProduct(int[] nums) {
        maxProductHelper(nums, 0);
        return max;
    }


    @Test
    public void testMaxProductDp() {
        int[] arr = arr(-5, 2, 4, 1, -2, 2, -6, 3, -1, -1, -1, -2, -3, 5, 1, -3, -4, 2, -4, 6, -1, 5);
        int[] arr2 = arr(0, -1, 4, -4, 5, -2, -1, -1, -2, -3, 0, -3, 0, 1, -1, -4, 4, 6, 2, 3, 0, -5, 2, 1, -4, -2, -1, 3, -4, -6, 0, 2, 2, -1, -5, 1, 1, 5, -6, 2, 1, -3, -6, -6, -3, 4, 0, -2, 0, 2);
        System.out.println(maxProduct(arr2));

        System.out.println(Arrays.stream(arr).reduce(1, (x, acc) -> x * acc));
    }

    public int[] sortedSquares(int[] nums) {
        int[] res = new int[nums.length];
        if (nums.length == 1) {
            return new int[]{nums[0] * nums[0]};
        }

        int right = 1;

        while (right < nums.length && nums[right] < 0) {
            right++;
        }

        int left = right - 1;

        int counter = 0;
        while (left >= 0 && right < nums.length) {
            if (Math.abs(nums[left]) < Math.abs(nums[right])) {
                res[counter++] = nums[left] * nums[left--];
            } else {
                res[counter++] = nums[right] * nums[right++];
            }
        }

        while (left >= 0) {
            res[counter++] = nums[left] * nums[left--];
        }

        while (right < nums.length) {
            res[counter++] = nums[right] * nums[right++];
        }

        System.out.println(Arrays.toString(res));

        return res;
    }


    @Test
    public void sortedSquares() {
        sortedSquares(arr(-5, -3, -2, -1));
        sortedSquares(arr(-4, -1, 0, 2, 3));
    }

    private boolean isPalindrome(char[] str, int start, int end) {

        int left = start;
        int right = end - 1;

        while (left <= right) {
            if (str[left] != str[right]) {
                return false;
            } else {
                left++;
                right--;
            }
        }
        return true;
    }

    private String createString(char[] str, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end; i++) {
            sb.append(str[i]);
        }
        return sb.toString();
    }

    private void backtrackPartition(char[] all, int start, List<String> curr, List<List<String>> res) {
        if (start >= all.length) {
            res.add(new ArrayList<>(curr));
            return;
        }
        for (int i = start; i < all.length; i++) {
            if (isPalindrome(all, start, i + 1)) {
                curr.add(createString(all, start, i + 1));
                backtrackPartition(all, i + 1, curr, res);
                curr.remove(curr.size() - 1);
            }
        }
    }

    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        char[] str = s.toCharArray();
        backtrackPartition(str, 0, new ArrayList<>(), res);

        return res;
    }

    @Test
    public void partitionTest() {
        System.out.println(partition("aab"));
        //System.out.println(isPalindrome("aaab".toCharArray(), 0, 4));
    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0 || nums[i] > n) {
                nums[i] = n + 1;
            }
        }

        for (int i = 0; i < nums.length; i++) {
            int index = Math.abs(nums[i]) - 1;
            if (index == n) continue;
            if (nums[index] > 0) nums[index] = -nums[index];

        }

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }

        return n + 1;
    }

    @Test
    public void testBackSpace() {
        System.out.println(firstMissingPositive(arr(1, 1)));
        System.out.println(firstMissingPositive(arr(2, 3)));
        System.out.println(firstMissingPositive(arr(1, 2, 0)));
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;

        int left = 0, right = m * n - 1;

        while (left <= right) {
            int pivot = left + (right - left) / 2;
            int midVal = matrix[pivot / n][pivot % n];
            if (midVal == target) {
                return true;
            }
            if (target > midVal) {
                left = pivot + 1;
            }
            if (target < midVal) {
                right = pivot - 1;
            }
        }

        return false;
    }

    @Test
    public void testSearchMatrix() {
        int[][] arr3 = new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}};

        System.out.println(searchMatrix(arr3, 11));
    }

    private void pathSumTraverse(TreeNode root, int targetSum, List<Integer> lst, List<List<Integer>> res) {
        if (root.left == null && root.right == null) {
            if (targetSum == root.val) {
                lst.add(root.val);
                res.add(new ArrayList<>(lst));
                lst.removeLast();
            }
        } else if (root.left == null) {
            lst.add(root.val);
            pathSumTraverse(root.right, targetSum - root.val, lst, res);
            lst.removeLast();
        } else if (root.right == null) {
            lst.add(root.val);
            pathSumTraverse(root.left, targetSum - root.val, lst, res);
            lst.removeLast();
        } else {
            lst.add(root.val);
            pathSumTraverse(root.left, targetSum - root.val, lst, res);
            lst.removeLast();
            lst.add(root.val);
            pathSumTraverse(root.right, targetSum - root.val, lst, res);
            lst.removeLast();
        }

    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        pathSumTraverse(root, targetSum, new ArrayList<>(), res);
        return res;
    }

    @Test
    public void testPathSum() {

        TreeNode root2 = new TreeNode(-2);

        // Level 1
        root2.left = null;
        root2.right = new TreeNode(-3);

        TreePrinter.print(root2);

        System.out.println(pathSum(root2, -5));

        TreeNode root = new TreeNode(5);

        // Level 1
        root.left = new TreeNode(4);
        root.right = new TreeNode(8);

        // Level 2
        root.left.left = new TreeNode(11);
        root.left.right = null;
        root.right.left = new TreeNode(13);
        root.right.right = new TreeNode(7);

        TreePrinter.print(root);

        System.out.println(pathSum(root, 20));


    }

    private void traverse(TreeNode root) {
        if (root == null) return;

        traverse(root.left);
        System.out.println(root.val);
        traverse(root.right);
    }

    private TreeNode rec(int[] preorder, int s1, int e1, int[] inorder, int s2, int e2) {
        if (s1 > e1 || s2 > e2) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[s1]);

        int inRootIndex = s2;
        while (inorder[inRootIndex] != preorder[s1]) {
            inRootIndex++;
        }
        int leftTreeSize = inRootIndex - s2;

        root.left = rec(preorder, s1 + 1, s1 + leftTreeSize, inorder, s2, inRootIndex - 1);
        root.right = rec(preorder, s1 + leftTreeSize + 1, e1, inorder, inRootIndex + 1, e2);

        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return rec(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    @Test
    public void testInorderPostorder() {
        TreeNode root = new TreeNode(1);

        // Level 1
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);

        // Level 2
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);

        TreePrinter.print(root);
        traverse(root);

        TreePrinter.print(buildTree(arr(3, 9, 20, 15, 7), arr(9, 3, 15, 20, 7)));
        TreePrinter.print(buildTree(arr(1, 2), arr(2, 1)));
    }

}
