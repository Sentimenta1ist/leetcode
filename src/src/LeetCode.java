package src;

import org.junit.jupiter.api.Test;
import src.structures.ListNode;
import src.structures.Node;
import src.structures.TreeNode;
import src.structures.TreePrinter;

import java.util.*;
import java.util.stream.Collectors;

import static src.structures.Node.printGraph;

public class LeetCode {

    private int[] arr(int... nums) {
        return nums;
    }

    private String[] arr(String... nums) {
        return nums;
    }

    private char[] arr(char... nums) {
        return nums;
    }

    public static int[][] matrix(String input) {
        // Remove the outer brackets
        input = input.substring(1, input.length() - 1);

        // Split into rows
        String[] rows = input.split("\\],\\[");

        // Determine the size of the 2D array
        int[][] result = new int[rows.length][];

        // Process each row
        for (int i = 0; i < rows.length; i++) {
            // Remove any remaining brackets and split by commas
            rows[i] = rows[i].replace("[", "").replace("]", "");
            String[] numbers = rows[i].split(",");

            // Convert strings to integers and populate the row
            result[i] = new int[numbers.length];
            for (int j = 0; j < numbers.length; j++) {
                result[i][j] = Integer.parseInt(numbers[j]);
            }
        }

        return result;
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

    public static void printMatrix(char[][] matrix) {
        System.out.println();
        for (char[] rows : matrix) {
            for (char element : rows) {
                System.out.printf("%4c", element);
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

    public boolean canPartition2(int[] nums) {
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
    public void canPartition2() {
        System.out.println(canPartition2(new int[]{1, 5, 11, 5}));
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

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        ListNode curr = head;
        int size = 1;
        while (curr.next != null) {
            curr = curr.next;
            size++;
        }

        int end = size - k % size;
        if (end == size) return head;

        ListNode newEnd = head;
        while (newEnd.next != null && end > 1) {
            newEnd = newEnd.next;
            end--;
        }
        ListNode newHead = newEnd.next;
        newEnd.next = null;
        curr.next = head;

        return newHead;
    }

    @Test
    public void testRotateList() {
        ListNode node = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5, null)))));

        ListNode.print(rotateRight(node, 12));

        ListNode node2 = new ListNode(1, new ListNode(2, null));

        ListNode.print(rotateRight(node2, 3));

        ListNode node3 = new ListNode(1, null);

        ListNode.print(rotateRight(node3, 10));

        ListNode.print(rotateRight(null, 10));
    }


    public int solution(String[] plan) {
        int[][] arr = transform(plan);
        int count = 0;

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                if (arr[i][j] == 2) { // If it's a dirty cell
                    dfs(arr, i, j); // Clean all room and build walls for not returning here :)
                    count++;
                }
            }
        }
        return count;
    }

    private void dfs(int[][] arr, int i, int j) {
        if (i < 0 || j < 0 || i >= arr.length || j >= arr[0].length || arr[i][j] == 0) {
            return; // if wall or out of range
        }
        arr[i][j] = 0;
        dfs(arr, i + 1, j);
        dfs(arr, i, j + 1);
        dfs(arr, i, j - 1);
        dfs(arr, i - 1, j);
    }

    // Method for transforming String[] to int[][]
    private int[][] transform(String[] plan) {
        int[][] arr = new int[plan.length][plan[0].length()];

        for (int i = 0; i < plan.length; i++) {
            for (int j = 0; j < plan[0].length(); j++) {
                char c = plan[i].charAt(j);
                if (c == '#') {
                    arr[i][j] = 0;  // Wall
                } else if (c == '.') {
                    arr[i][j] = 1;  // Clean floor
                } else if (c == '*') {
                    arr[i][j] = 2;  // Dirty floor
                }
            }
        }
        return arr;
    }
    // Time: Big(O) = M * N
    // Space: Big(O) = M * N
    // M -> plan length
    // N -> plan[0].length


    // Time: Big(O) = (M * H);
    // M - skills size
    // H -> height of tree (longest path from root to skill)
    //
    // Space: Big(O) = N
    // N - number skills in the tree (tree size)
    public int solution(int[] tree, int[] skills) {
        final Set<Integer> learned = new HashSet<>();

        for (int skill : skills) {
            int curr = skill;

            // Traverse to the bottom of the tree learning every skill on the path
            while (!learned.contains(curr)) {
                learned.add(curr);
                curr = tree[curr];
            }
        }

        // all learned skill due traversing skills array
        return learned.size();
    }


    @Test
    public void testSolution1() {
        System.out.println(solution(arr(0, 0, 1, 1), arr(2)));
        System.out.println(solution(arr(0, 0, 0, 0, 2, 3, 3), arr(2, 5, 6)));
        System.out.println(solution(arr(0, 3, 0, 0, 5, 0, 5), arr(4, 2, 6, 1, 0)));
        System.out.println(solution(arr(0), arr(0)));
    }

    private void dfs(Node node, Map<Node, Node> map) {
        if (map.containsKey(node)) {
            return;
        }
        Node newNode = new Node(node.val);
        map.put(node, newNode);
        for (Node neigh : node.neighbors) {
            if (!map.containsKey(neigh)) {
                dfs(neigh, map);

            }
            map.get(neigh).neighbors.add(newNode);
        }
    }

    public Node cloneGraph(Node node) {
        Map<Node, Node> map = new HashMap<>();
        dfs(node, map);
        return map.get(node);
    }

    @Test
    public void testCloneGraph() {
        // Example graph creation
        Node node1 = new Node(1);
        Node node2 = new Node(2);
        Node node3 = new Node(3);
        //Node node4 = new Node(4);
        //Node node5 = new Node(5);

        // Create graph connections
        node1.neighbors.add(node2);
        node1.neighbors.add(node3);

        node2.neighbors.add(node1);
        node2.neighbors.add(node3);
        // node2.neighbors.add(node3);

        node3.neighbors.add(node1);
        node3.neighbors.add(node2);
        // node3.neighbors.add(node5);

        //   node4.neighbors.add(node5);

        // node5.neighbors.add(node3);

        printGraph(node1);
        System.out.println("----");
        printGraph(cloneGraph(node1));
    }

    boolean ans = true;

    public boolean hasCycle(List<List<Integer>> graph, int from, int[] visited) {
        visited[from] = 1;

        for (int to : graph.get(from)) {
            if (visited[to] == 1) {
                return true;
            }
            if (visited[to] == 0 && hasCycle(graph, to, visited)) {
                return true;
            }
        }
        visited[from] = 2;
        return false;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {

        List<List<Integer>> graph = new ArrayList<>(numCourses);
        for (int i = 0; i < numCourses; i++)
            graph.add(new ArrayList<>());
        for (int[] prerequisite : prerequisites)
            graph.get(prerequisite[0]).add(prerequisite[1]);

        int[] visited = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            if (visited[i] == 0 && hasCycle(graph, i, visited)) {
                return false;
            }
        }

        return true;
    }

    @Test
    public void testCanFinish() {
        int numCourses1 = 4;
        int[][] prerequisites1 = {{1, 0}, {2, 0}, {3, 0}};
        System.out.println(canFinish(numCourses1, prerequisites1));

        int numCourses2 = 2;
        int[][] prerequisites2 = {{1, 0}, {0, 1}};
        System.out.println(canFinish(numCourses2, prerequisites2));

        int numCourses4 = 4;
        int[][] prerequisites4 = {{1, 0}, {2, 0}, {3, 2}, {3, 1}};
        System.out.println(canFinish(numCourses4, prerequisites4));
    }


    public int trap(int[] height) {
        int res = 0, left = 0, right = 0, n = height.length;

        while (left < n - 1 && right < n) {
            right = left + 1;
            int tmp = height[left];
            while (right < n) {
                if (height[right] >= height[left]) {
                    res += ((right - left) * height[left]) - tmp;
                    left = right;
                    break;
                }
                tmp += height[right++];

            }
        }
        right--;
        while (right > left) {
            int left2 = right - 1;
            int tmp = height[right];

            while (left2 >= left) {
                if (height[left2] >= height[right]) {
                    res += ((right - left2) * height[right]) - tmp;
                    right = left2;
                    break;
                }

                tmp += height[left2--];

            }

        }

        return res;
    }

    @Test
    public void testTrap() {
        System.out.println(trap(arr(3, 0, 2))); // 2
        System.out.println(trap(arr(3, 2, 1, 2, 1))); // 1
        System.out.println(trap(arr(0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1))); // 6
        System.out.println(trap(arr(2, 1, 0, 1, 3, 2, 1, 2, 1))); // 5
        System.out.println(trap(arr(0, 0, 0, 0))); // 5
        System.out.println(trap(arr(0, 1, 0))); // 5
        System.out.println(trap(arr(1, 2, 3, 4, 5))); // 5
        System.out.println(trap(arr(0, 1, 0, 2, 0, 1, 0, 2, 0))); // 5
    }

    public int characterReplacement2(String s, int k) {
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

    @Test
    public void testCharReplacement() {
        System.out.println(characterReplacement2("AABA", 0));
        System.out.println(characterReplacement2("BABA", 0));
        System.out.println(characterReplacement2("AAAA", 0));
        System.out.println(characterReplacement2("AABABAAAAA", 1));
        System.out.println(characterReplacement2("ABCDE", 1));
        System.out.println(characterReplacement2("ABBB", 2));

    }

    public static boolean isNumeric(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            double d = Double.parseDouble(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    private Integer eval(String op, Integer op1, Integer op2) {
        return switch (op) {
            case "/" -> op1 / op2;
            case "+" -> op1 + op2;
            case "-" -> op1 - op2;
            case "*" -> op1 * op2;
            default -> 0;
        };
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();

        for (String s : tokens) {
            switch (s) {
                case "+" -> stack.add(stack.pop() + stack.pop());
                case "-" -> {
                    int a = stack.pop();
                    int b = stack.pop();
                    stack.add(b - a);
                }
                case "/" -> {
                    int a = stack.pop();
                    int b = stack.pop();
                    stack.add(b / a);
                }
                case "*" -> stack.add(stack.pop() * stack.pop());
                default -> stack.add(Integer.parseInt(s));
            }
        }
        return stack.peek();
    }

    public int minSubarray(int[] nums, int p) {
        long sum = 0;
        for (int num : nums) {
            sum += num;
        }

        long remainder = sum % p;

        if (remainder == 0) return 0;
        HashMap<Long, Integer> prefixSumMap = new HashMap<>();
        prefixSumMap.put(0L, -1);

        long currentSum = 0;
        int minLength = nums.length;

        for (int i = 0; i < nums.length; i++) {
            currentSum += nums[i];
            long modValue = (currentSum % p + p) % p;
            long targetMod = (modValue - remainder + p) % p;

            if (prefixSumMap.containsKey(targetMod)) {
                minLength = Math.min(minLength, i - prefixSumMap.get(targetMod));
            }

            prefixSumMap.put(modValue, i);
        }

        return minLength == nums.length ? -1 : minLength;
    }

    @Test
    public void testRPN() {
        System.out.println(evalRPN(arr("2", "1", "+", "3", "*"))); // 9
        System.out.println(evalRPN(arr("4", "13", "5", "/", "+"))); // 6
        System.out.println(evalRPN(arr("10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"))); // 22
        System.out.println(minSubarray(arr(1000000000, 1000000000, 1000000000), 3));
        System.out.println(minSubarray(arr(26, 19, 11, 14, 18, 4, 7, 1, 30, 23, 19, 8, 10, 6, 26, 3), 26));

    }


    PriorityQueue<Integer> heap;
    int k;

    public void asd(int k, int[] nums) {
        this.k = k;
        heap = new PriorityQueue<>(k);

        for (int num : nums) {
            heap.add(num);
        }
//        while(heap.size() > k) {
//            heap.poll();
//        }
    }

    public int add(int val) {
        heap.add(val);
        if (heap.size() > k) {
            heap.poll();
        }
        return heap.peek();
    }

    public int[] dailyTemperatures(int[] arr) {

        int n = arr.length;
        int[] res = new int[arr.length];
        for (int i = n - 1; i >= 0; i--) {

            int counter = 0;
            for (int j = i; j < n && arr[i] <= arr[j]; j++) {
                counter++;
            }

            res[arr.length - i - 1] = counter;
        }

        return res;

    }

    public int[] topKFrequent(int[] nums, int k) {
        int[] res = new int[k];

        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        System.out.println(map);

        PriorityQueue<int[]> heap = new PriorityQueue<>((x, y) -> x[1] - y[1]);

        for (Integer key : map.keySet()) {
            heap.add(new int[]{key, map.get(key)});
            if (heap.size() > k) {
                heap.poll();
            }
        }

        int counter = 0;
        for (int[] pair : heap) {
            res[counter++] = pair[0];
        }

        return res;
    }


    public long dividePlayers(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        int weak = Integer.MAX_VALUE, strong = Integer.MIN_VALUE;
        long res = 0;

        for (int num : arr) {
            weak = Math.min(weak, num);
            strong = Math.max(strong, num);
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        int average = weak + strong;

        for (int player1 : arr) {
            if (map.get(player1) == 0) continue;
            int player2 = average - player1;
            if (map.getOrDefault(player2, 0) == 0) {
                return -1;
            } else {
                map.put(player1, map.get(player1) - 1);
                map.put(player2, map.get(player2) - 1);
                res += (long) player1 * (long) player2;
            }
        }
        return res;
    }

    public String minWindow(String s, String t) {
        int left = 0, right = 0;
        boolean found = false;
        int minLeft = 0;
        int minRight = s.length() + 1; // Initialize to s.length() + 1 to check if it changes

        Map<Character, Integer> source = new HashMap<>();
        Map<Character, Integer> target = new HashMap<>();

// Initialize target map with character counts from t
        for (Character c : t.toCharArray()) {
            target.put(c, target.getOrDefault(c, 0) + 1);
        }

// Initialize the source map with 0 counts for all characters in t
        for (Character c : target.keySet()) {
            source.put(c, 0);
        }

        char[] arr = s.toCharArray();
        int need = target.size(); // Number of unique characters we need to match
        int have = 0; // Number of unique characters we've matched so far

        while (right < arr.length) {
            // If the current character in `s` is part of `t`, update source map
            if (target.containsKey(arr[right])) {
                source.put(arr[right], source.get(arr[right]) + 1);
                // Only increment `have` if we've matched the count exactly with target
                if (source.get(arr[right]).equals(target.get(arr[right]))) {
                    have++;
                }
            }

            // When all characters are matched, try to shrink the window from the left
            while (have == need) {
                found = true;
                // Check if the current window is the smallest
                if (right - left + 1 < minRight - minLeft) {
                    minLeft = left;
                    minRight = right + 1;
                }
                // If the character at `left` is part of `t`, update source map
                if (target.containsKey(arr[left])) {
                    source.put(arr[left], source.get(arr[left]) - 1);
                    // If the count goes below what's required, reduce `have`
                    if (source.get(arr[left]) < target.get(arr[left])) {
                        have--;
                    }
                }
                left++; // Shrink window from the left
            }

            right++; // Expand window by moving `right` forward
        }

// Return the result based on whether a valid window was found
        return found ? s.substring(minLeft, minRight) : "";

    }

    boolean rotting = true;

    private void rottCell(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1) {
            return;
        }
        rotting = true;
        grid[i][j] = 2;
    }

    private void rottPlace(int[][] grid, int i, int j) {
        rottCell(grid, i, j + 1);
        rottCell(grid, i + 1, j);
        rottCell(grid, i - 1, j);
        rottCell(grid, i, j - 1);
    }

    private void holera(int[][] grid, List<int[]> places) {
        for (int[] place : places) {
            rottPlace(grid, place[0], place[1]);
        }
    }

    public int orangesRotting(int[][] grid) {
        int counter = 0;

        while (rotting) {
            rotting = false;

            List<int[]> rotted = new ArrayList<>();

            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == 2) {
                        rotted.add(new int[]{i, j});
                    }
                }
            }

            holera(grid, rotted);
            counter++;
        }

        for (int[] ints : grid) {
            for (int j = 0; j < grid[0].length; j++) {
                if (ints[j] == 1) {
                    return -1;
                }
            }
        }

        return counter - 1;
    }

    public String serialize(TreeNode node) {
        StringBuilder sb = new StringBuilder();
        serialize(node, sb);
        return sb.toString();
    }

    public void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("X");
            return;
        }
        sb.append(root.val);
        if (root.left != null || root.right != null) {
            sb.append("(");
            serialize(root.left, sb);
            sb.append(")(");
            serialize(root.right, sb);
            sb.append(")");
        }
    }

    private Integer getFirst(String data) {
        if (data.isEmpty()) return null;
        if (data.charAt(0) == 'X') return null;
        int index = data.indexOf('(');
        return Integer.parseInt(index == -1 ? data : data.substring(0, index));
    }

    private String getLeft(String data) {

        int index = data.indexOf('(');
        if (index == -1) return "";
        int counter = 0;
        for (int i = index; i < data.length(); i++) {
            if (data.charAt(i) == '(') counter++;
            if (data.charAt(i) == ')') counter--;
            if (counter == 0) return data.substring(index + 1, i);
        }
        return "";
    }

    private String getRight(String data) {
        char[] arr = data.toCharArray();
        int counter = 0;
        if (arr[arr.length - 1] != ')') return "";
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == ')') counter++;
            if (arr[i] == '(') counter--;
            if (counter == 0) return data.substring(i + 1, arr.length - 1);
        }

        return "";
    }

    private TreeNode rec(String data, TreeNode root) {
        if (data.isEmpty() || getFirst(data) == null) {
            return null;
        }
        root.val = getFirst(data);
        root.left = rec(getLeft(data), new TreeNode());
        root.right = rec(getRight(data), new TreeNode());

        return root;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        return rec(data, new TreeNode());
    }

    private boolean dfs(char[][] board, int i, int j) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
            return false;
        }
        if (board[i][j] != 'O') {
            return true;
        }
        board[i][j] = '|';
        return dfs(board, i, j + 1) && dfs(board, i + 1, j) && dfs(board, i - 1, j) && dfs(board, i, j - 1);
    }

    private void fill(char[][] board, int i, int j, char s, char t) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != s) {
            return;
        }

        board[i][j] = t;
        fill(board, i, j + 1, s, t);
        fill(board, i + 1, j, s, t);
        fill(board, i - 1, j, s, t);
        fill(board, i, j - 1, s, t);
    }

    public void solve(char[][] board) {


        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O')
                fill(board, i, 0, 'O', '|');

            if (board[i][board[0].length - 1] == 'O')
                fill(board, i, board[0].length - 1, 'O', '|');
        }
        for (int j = 1; j < board[0].length - 1; j++) {
            if (board[0][j] == 'O')
                fill(board, 0, j, 'O', '|');
            fill(board, board.length - 1, j, 'O', '|');
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 'O') {
                    fill(board, i, j, 'O', 'X');
                }
            }
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '|') {
                    board[i][j] = 'O';
                }
            }
        }
    }

    @Test
    public void testIslands() {
        char[][] arr = new char[][]{
                {'X', 'X', 'X', 'X'},
                {'X', 'O', 'O', 'X'},
                {'X', 'X', 'O', 'X'},
                {'X', 'X', 'O', 'X'},
                {'X', 'O', 'X', 'X'}};
        solve(arr);

        printMatrix(arr);

        char[][] arr2 = new char[][]{
                {'O', 'O', 'O'},
                {'O', 'X', 'O'},
                {'O', 'O', 'O'}};
        solve(arr2);

        printMatrix(arr2);
    }

    private void removeNode(List<List<Integer>> graph, Integer node) {
        for (List<Integer> integers : graph) {
            integers.remove(node);
        }
    }

    boolean cycle = false;

    public void topologicalSort(List<List<Integer>> graph, Integer from, int[] visited, List<Integer> res) {
        visited[from] = 1;

        for (Integer to : graph.get(from)) {
            if (visited[to] == 1) {
                cycle = true;
                return;
            }
            if (visited[to] == 0) {
                topologicalSort(graph, to, visited, res);
            }

        }
        visited[from] = 2;
        res.add(from);
    }

    public int[] findOrder(int n, int[][] arr) {
        List<List<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());

        for (int[] pair : arr)
            graph.get(pair[0]).add(pair[1]);

        List<Integer> res = new ArrayList<>();

        int[] visited = new int[n];
        for (int i = 0; i < visited.length; i++) {
            if (visited[i] == 0) {
                cycle = false;
                topologicalSort(graph, i, visited, res);
                if (cycle) return new int[]{};
            }

        }

        return res.stream().mapToInt(i -> i).toArray();
    }

    @Test
    public void testFindOrder() {

        System.out.println(Arrays.toString(
                findOrder(4, new int[][]{
                        {1, 0},
                        {2, 0},
                        {3, 1},
                        {3, 2}}))); // 0 2 1 3

        System.out.println(Arrays.toString(
                findOrder(2, new int[][]{
                        {0, 1}}))); // 0 1

        System.out.println(Arrays.toString(
                findOrder(2, new int[][]{
                        {1, 0}}))); // 0 1

    }

    int from = -1;
    int to = -1;

    private boolean cycleSearch(List<List<Integer>> graph, int prev, int curr, int[] visited) {
        visited[curr] = 1;

        for (Integer to : graph.get(curr)) {
            if (visited[to] == 1 && to != prev) {
                return true;
            }
            if (visited[to] == 0 && cycleSearch(graph, curr, to, visited)) {
                return true;
            }
        }

        visited[curr] = 2;
        return false;
    }

    public int[] findRedundantConnection(int[][] edges) {
        List<List<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < edges.length; i++)
            graph.add(new ArrayList<>());

        for (int[] pair : edges) {
            graph.get(pair[1] - 1).add(pair[0] - 1);
            graph.get(pair[0] - 1).add(pair[1] - 1);

            int[] visited = new int[edges.length];
            if (cycleSearch(graph, -1, pair[1] - 1, visited)) {
                return pair;
            }
        }


        return new int[]{0, 0};
    }

    @Test
    public void testFindRedundantConnection() {


        System.out.println(Arrays.toString(
                findRedundantConnection(new int[][]{
                        {1, 2},
                        {1, 3},
                        {2, 3}}))); // 0 2 1 3
    }

    private void dfs(char[] digits, int start, StringBuilder comb, List<String> res, Map<Character, String> values) {
        if (start >= digits.length) {
            res.add(comb.toString());
            return;
        }
        String str = values.get(digits[start]);
        for (int i = 0; i < str.length(); i++) {
            comb.append(str.charAt(i));
            dfs(digits, start + 1, comb, res, values);
            comb.deleteCharAt(comb.length() - 1);
        }
    }

    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();

        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");

        dfs(digits.toCharArray(), 0, new StringBuilder(), res, map);

        return res;

    }

//    int minDiff = Integer.MAX_VALUE;

    private static final int[] dx = {0, 0, 1, -1};
    private static final int[] dy = {1, -1, 0, 0};


    public int minimumEffortPath(int[][] height) {
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));

        int[][] visited = new int[height.length][height[0].length];
        for (int i = 0; i < height.length; i++) {
            Arrays.fill(visited[i], Integer.MAX_VALUE); // Initialize with max value
        }

        queue.offer(arr(0, 0, 0));
        visited[0][0] = 0;

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                int[] node = queue.remove();
                int x = node[0];
                int y = node[1];
                int curr = height[x][y];
                int maxDiff = node[2];

                for (int d = 0; d < 4; d++) {
                    int nx = dx[d] + x;
                    int ny = dy[d] + y;


                    if (nx < 0 || ny < 0 || nx >= height.length || ny >= height[0].length) {
                        continue;
                    }
                    int neigh = height[nx][ny];
                    int currDiff = Math.max(Math.abs(neigh - curr), maxDiff);

                    if (visited[nx][ny] > currDiff) {
                        visited[nx][ny] = currDiff;
                        queue.add(arr(nx, ny, currDiff));
                    }
                }
            }
        }

        return visited[height.length - 1][height[0].length - 1];
    }

    @Test
    public void testMinimumEffort() {
        System.out.println(minimumEffortPath(matrix("[[1,2,2]," +
                "[3,8,2]," +
                "[5,3,5]]")));
        System.out.println(minimumEffortPath(matrix("[[1,2,3],[3,8,4],[5,3,5]]]")));
        System.out.println(minimumEffortPath(matrix("[[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]")));
        System.out.println(minimumEffortPath(matrix("[[8,3,2,5,2,10,7,1,8,9],[1,4,9,1,10,2,4,10,3,5],[4,10,10,3,6,1,3,9,8,8],[4,4,6,10,10,10,2,10,8,8],[9,10,2,4,1,2,2,6,5,7],[2,9,2,6,1,4,7,6,10,9],[8,8,2,10,8,2,3,9,5,3],[2,10,9,3,5,1,7,4,5,6],[2,3,9,2,5,10,2,7,1,8],[9,10,4,10,7,4,9,3,1,6]]")));

    }

    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        StringBuilder commonPrefix = new StringBuilder();
        int start1 = 0;
        int end1 = sentence1.length() - 1;
        int start2 = 0;
        int end2 = sentence2.length() - 1;

        while (start1 <= end1 && start2 <= end2) {
            char element = sentence1.charAt(start1);
            if (element == sentence2.charAt(start2)) {
                commonPrefix.append(element);
                start1++;
                start2++;
            } else {
                break;
            }
        }

        StringBuilder commonSuffix = new StringBuilder();
        while (start1 <= end1 && start2 <= end2) {
            char element = sentence1.charAt(end1);
            if (element == sentence2.charAt(end2)) {
                commonSuffix.insert(0, element);
                end1--;
                end2--;
            } else {
                break;
            }
        }


        if (commonPrefix.isEmpty() && commonSuffix.isEmpty()) return false;
        String center;
        if (sentence1.length() > sentence2.length()) {
            center = sentence1.substring(start1, end1);
            return sentence1.equals(commonPrefix + center + " " + commonSuffix);
        } else {
            center = sentence2.substring(start2, end2);
            return sentence2.equals(commonPrefix + center + " " + commonSuffix);
        }
    }

    @Test
    public void testCommonts() {
        System.out.println(areSentencesSimilar("My name is Haley", "My Haley"));
        System.out.println(areSentencesSimilar("Frogs are cool", "Frog cool"));
        System.out.println(areSentencesSimilar("Frog are cool", "Frog cool"));
        System.out.println(areSentencesSimilar("A lot of words", "of"));
    }

    public int minLength(String s) {
        int currSize = s.length();
        boolean removed = true;
        while (removed) {
            removed = false;
            s = s.replace("AB", "");
            s = s.replace("CD", "");
            if (s.length() < currSize) {
                removed = true;
                currSize = s.length();
            }
        }

        return currSize;
    }


    @Test
    public void testMinLength() {
        System.out.println(minLength("ABFCACDB"));
    }

    public int maxProfit2(int[] arr) {
        int min = arr[0];
        int profit = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] >= arr[i - 1]) {
                profit += arr[i] - min;
            }
            min = arr[i];
        }

        return profit;
    }

    @Test
    public void testMaxProfit() {
        System.out.println(maxProfit2(arr(7, 1, 5, 3, 6, 4)));
        System.out.println(maxProfit2(arr(1, 2, 3, 4, 5)));
    }

    public int searchInsert(int[] arr, int target) {
        int l = 0;
        int r = arr.length - 1;

        while (l <= r) {
            int pivot = r - l / 2 + l;
//            System.out.println(pivot);
//            System.out.println(arr[pivot]);
            if (target == arr[pivot]) {
                return pivot;
            }
            if (arr[pivot] > target) {
                r = pivot - 1;
            } else {
                l = pivot + 1;
            }
        }

        return l;
    }

    @Test
    public void testInsertPlace() {
        System.out.println(searchInsert(arr(1, 3, 5, 6), 5));
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> res = new ArrayList<>();

        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> (b[0] + b[1]) - (a[0] + a[1]));

        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                if (!queue.isEmpty() && queue.size() >= k) {
                    int[] pair = queue.peek();
                    if (pair[0] + pair[1] <= nums1[i] + nums2[j]) {
                        break;
                    }
                }
                queue.add(new int[]{nums1[i], nums2[j]});
                if (queue.size() > k) {
                    queue.poll();
                }
            }
        }

        for (int[] pair : queue) {
            List<Integer> pairList = new ArrayList<>(2);
            pairList.add(pair[0]);
            pairList.add(pair[1]);
            res.add(pairList);
        }

        return res;
    }

    @Test
    public void testSmallestPairs() {
        System.out.println(kSmallestPairs(arr(1, 1, 2), arr(1, 2, 3), 2));
        //System.out.println(kSmallestPairs(arr(1,7,11), arr(2,4,6), 3));
    }

    public static boolean canFullyFillKnapsack(int[] counts, int capacity) {
        // Create a DP array to store whether a sum can be achieved
        boolean[] dp = new boolean[capacity + 1];

        // We can always achieve a sum of 0 (by taking no elements)
        dp[0] = true;

        // Iterate over each weight (the index of the array) and its corresponding count (the value at that index)
        for (int weight = 1; weight < counts.length; weight++) {
            int count = counts[weight];

            // Traverse the dp array backwards to avoid overwriting previous results
            for (int i = capacity; i >= 0; i--) {
                // Use each weight up to 'count' times
                for (int j = 1; j <= count && i - j * weight >= 0; j++) {
                    dp[i] = dp[i] || dp[i - j * weight];
                }
            }
        }

        // Return whether we can achieve the sum equal to capacity
        return dp[capacity];
    }

    public boolean canPartition(int[] nums) {
        int[] weights = new int[101];
        int sum = 0;
        for (int num : nums) {
            weights[num] = weights[num] + 1;
            sum += num;
        }

        if (sum % 2 != 0) return false;

        return canFullyFillKnapsack(weights, sum / 2);
    }

    @Test
    public void testCanPartition() {
        System.out.println(canPartition(arr(1, 5, 11, 5)));
    }

    public int hammingWeight(int n) {
        long pow = 1;

        int counter = 0;

        while (pow <= n) {
            if ((n & pow) != 0) {
                counter++;
            }
            pow *= 2;
        }

        return counter;
    }

    @Test
    public void testHammingWeight() {
        System.out.println(hammingWeight(2147483645)); // 1011
    }

    int lev = Integer.MAX_VALUE;

    public void traverse(TreeNode root, int target, int level, int k, List<Integer> res) {
        if (root == null) {
            return;
        }
        if (root.val == target) {
            lev = 0;
        }

        lev++;
        traverse(root.left, target, level + 1, k, res);
        if (Math.abs(lev) == k) {
            res.add(root.val);
        }
        traverse(root.right, target, level + 1, k, res);

        lev--;

        if (lev == 0) {
            res.add(root.val);
        }
    }

    public List<Integer> distanceK(TreeNode root, int target, int k) {
        List<Integer> res = new ArrayList<>();


        traverse(root, target, 0, k, res);

        return res;
    }

    @Test
    public void testDistanceK() {
        String tree = "3(9(5(6)(2(7)(4)))(10))(1(0)(8))";
        TreeNode root = deserialize(tree);
        TreePrinter.print(root);

        System.out.println(distanceK(root, 5, 2));
    }

    int getMinimalChair(PriorityQueue<int[]> queue, int[] interval) {
        if (queue.isEmpty()) {
            queue.add(interval);
            return 0;
        } else {
            int counter = 0;
            for (int[] ints : queue) {
                if (ints[1] <= interval[0]) {
                    ints[0] = interval[0];
                    ints[1] = interval[1];
                    return counter;
                } else {
                    counter++;
                }
            }
            queue.add(interval);
            return queue.size() - 1;
        }
    }

    public int smallestChair(int[][] times, int targetFriend) {
        int res = 0;

        int targetArrivalTime = times[targetFriend][0];
        Map<int[], Integer> map = new HashMap<>();
        for (int i = 0; i < times.length; i++) {
            map.put(times[i], i);
        }

        Arrays.sort(times, (a, b) -> Integer.compare(a[0], b[0]));
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] - b[0]);

        for (int i = 0; i < times.length; i++) {
            int minChair = getMinimalChair(queue, times[i]);

            if (targetArrivalTime == times[i][0]) {
                return minChair;
            }
        }

        return 0;
    }

    @Test
    public void testSmallestChair() {
        int[][] arr = matrix("[[3,10],[1,5],[2,6]]");
        printMatrix(arr);
        System.out.println(smallestChair(arr, 0));
    }


    // Check if a number is prime
    private static boolean isPrime(long n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (long i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    // Generate all prime numbers up to a specified limit
    private static List<Long> generatePrimes(long limit) {
        List<Long> primes = new ArrayList<>();
        for (long num = 2; num <= limit; num++) {
            if (isPrime(num)) {
                primes.add(num);
            }
        }
        return primes;
    }

    // Find x for prime y such that x || (x + 1) = y
    private int[] findXForPrimeY(List<Integer> primes) {
        int[] res = new int[primes.size()];
        Arrays.fill(res, -1);
        int counter = 0;
        for (int y : primes) {

            for (int x = 0; x * x < y; x++) {
                if ((x | (x + 1)) == y) {
                    res[counter] = x;
                    break;  /// Return the first (smallest) solution found
                }
            }

            // Continue from higher values based on the structure of y
            for (int x = (y - 1); x >= 0; x--) {
                if ((x | (x + 1)) == y) {
                    res[counter] = x;
                    break;  /// Return the first (smallest) solution found
                }
            }

            counter++;
        }
        return res;
    }

    @Test
    public void testFindX() {
        System.out.println(Arrays.toString(findXForPrimeY(List.of(2, 3, 5, 7))));
        System.out.println(Arrays.toString(findXForPrimeY(List.of(11, 13, 31))));
        System.out.println(Arrays.toString(findXForPrimeY(List.of(178332559, 228675283, 19360877, 205565587, 438746107, 973570693, 355742033, 449983951, 214272103, 327283903, 523677797, 36361697, 795061931, 189520679, 499936331))));

    }

    public long maxKelements(int[] nums, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>(Comparator.reverseOrder());

        for (int num : nums) {
            heap.add(num);
        }

        long sum = 0;

        while (k > 0) {
            int max = heap.remove();
            sum += max;
            heap.offer((int) Math.ceil((double) max / 3));
            k--;
        }

        return sum;
    }

    @Test
    public void testMaxLElements() {
        System.out.println(maxKelements(arr(1, 10, 3, 3, 3), 3));
        System.out.println(maxKelements(arr(10, 10, 10, 10, 10), 5));
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();

        res.add(List.of(1));
        int size = 2;
        Integer[] prev = null;
        for (int i = 1; i < numRows; i++) {
            Integer[] level = new Integer[size];
            level[0] = 1;
            level[size - 1] = 1;
            for (int j = 1; j < size - 1; j++) {
                level[j] = prev[j - 1] + prev[j];
            }
            prev = level;
            res.add(List.of(level));
            size++;
        }

        return res;
    }

    @Test
    public void testPascal() {
        System.out.println(generate(5));
    }

    public long minimumSteps(String s) {
        char[] arr = s.toCharArray();
        long res = 0;
        int place = arr.length - 1;
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == '1') {
                res += place - i;
                place--;
            }
        }

        return res;
    }

    @Test
    public void testMinSteps() {
        System.out.println(minimumSteps("0011000010"));
        System.out.println(minimumSteps("101"));
        System.out.println(minimumSteps("0111"));
        System.out.println(minimumSteps("100"));
    }


    public boolean searchMatrix1(int[][] arr, int target) {
        int i = arr.length - 1;
        int j = arr[0].length - 1;

        while (i >= 0 && j >= 0) {
            if (arr[i][j] == target || arr[i][0] == target || arr[0][j] == target) {
                return true;
            }
            if (arr[i][0] > target) {
                i--;
            }
            if (arr[0][j] > target) {
                j--;
            }
            if (i >= 0 && j >= 0 && arr[i][0] < target && arr[0][j] < target) {
                for (int start = 0; start <= j; start++) {
                    if (arr[i][start] == target) {
                        return true;
                    }
                }
                for (int start = 0; start <= i; start++) {
                    if (arr[start][j] == target) {
                        return true;
                    }
                }
                i--;
                j--;
            }
        }
        return false;
    }

    @Test
    public void testBinarySearchMatr() {
        int[][] arr = matrix("[[1,3,5,7,9],[2,4,6,8,10],[11,13,15,17,19],[12,14,16,18,20],[21,22,23,24,25]]");
        printMatrix(arr);
        System.out.println(searchMatrix1(arr, 13));
    }


    public String reorganizeString(String s) {
        int[] arr = new int[26];

        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> b[0] - a[0]);

        for (int i = 0; i < s.length(); i++) {
            arr[s.charAt(i) - 97]++;

            if (arr[s.charAt(i) - 97] > (s.length() + 1) / 2) {
                return "";
            }
        }

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != 0) {
                queue.add(new int[]{arr[i], 'a' + i});
            }
        }

        StringBuilder sb = new StringBuilder();

        int[] prev = new int[]{0, 0};
        while (!queue.isEmpty()) {
            int[] top = queue.remove();
            sb.append((char) top[1]);
            top[0]--;
            if (prev[0] > 0) {
                queue.offer(prev);
            }
            prev = top;

        }
        return sb.toString();
    }

    @Test
    public void testReoStreings() {
        System.out.println(reorganizeString("aab"));
        System.out.println(reorganizeString("aaab"));
        System.out.println(reorganizeString("vvvlo"));
    }

    public String longestDiverseString(int a, int b, int c) {
        PriorityQueue<int[]> heap = new PriorityQueue<>((x1, x2) -> x2[0] - x1[0]);
        if (a > 0) heap.add(new int[]{a, 'a'});
        if (b > 0) heap.add(new int[]{b, 'b'});
        if (c > 0) heap.add(new int[]{c, 'c'});


        StringBuilder sb = new StringBuilder();
//        int[] prev = new int[]{0, 0};
        int counter = 0;
        while (!heap.isEmpty()) {
            int[] top = heap.poll();
            if (top[0] == 0) {
                continue;
            }
            if (counter > 2) {
                sb.append((char) top[1]);
                top[0]--;
                heap.add(top);
                counter++;
                continue;
            }


        }

        return sb.toString();
    }


    @Test
    public void testLongestDiverseString() {
        System.out.println(longestDiverseString(40, 5, 0));
        System.out.println(longestDiverseString(7, 1, 0));
    }

    public void rotate(int[][] arr) {
        int n = arr.length;
        for (int k = 0; k < n / 2; k++) {
            for (int i = k; i < n - 1 - k; i++) {
                int tmp = arr[i][k];
                arr[i][k] = arr[n - 1 - k][i];
                arr[n - 1 - k][i] = arr[n - 1 - i][n - 1 - k];
                arr[n - 1 - i][n - 1 - k] = arr[k][n - 1 - i];
                arr[k][n - 1 - i] = tmp;
            }
        }
    }

    @Test
    public void testRotate() {
        int[][] arr = matrix("[[1,2,3],[4,5,6],[7,8,9]]");
        printMatrix(arr);
        rotate(arr);
        printMatrix(arr);
    }

    private int openParents(StringBuilder s) {
        int counter = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') counter++;
            if (s.charAt(i) == ')') counter--;
            if (counter < 0) return counter;
        }
        return counter;
    }

    private void backtrack(int n, StringBuilder curr, List<String> res, int open, int closed) {
        if (curr.length() == n * 2) {
            res.add(curr.toString());
            return;
        }

        // Try adding an opening parenthesis if we haven't reached the limit
        if (open < n) {
            curr.append("(");
            backtrack(n, curr, res, open + 1, closed);
            curr.deleteCharAt(curr.length() - 1); // Undo the addition
        }

        // Try adding a closing parenthesis if it's valid (close < open)
        if (closed < open) {
            curr.append(")");
            backtrack(n, curr, res, open, closed + 1);
            curr.deleteCharAt(curr.length() - 1); // Undo the addition
        }
    }


    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        backtrack(n, new StringBuilder(), res, 0, 0);
        return res;
    }

    @Test
    public void testGeParen() {
        System.out.println(generateParenthesis(3));
    }

    int nQueens = 0;

    private void insertQueen(int[][] arr, int i, int j) {
        arr[i][j] = 1;
        for (int start = 0; start < arr.length; start++) {
            arr[start][j] = -1;
        }
        for (int start = j + 1; start < arr.length; start++) {
            arr[i][start] = -1;
        }

        for (int start = i + 1, start2 = j + 1; start < arr.length && start2 < arr.length; start++, start2++) {
            arr[start][start2] = -1;
        }


        for (int start = i - 1, start2 = j + 1; start >= 0 && start2 < arr.length; start++, start2++) {
            arr[start][start2] = -1;
        }
    }

    private void removeQueen(int[][] arr, int i, int j) {
        arr[i][j] = 0;
        for (int start = i + 1; start < arr.length; start++) {
            arr[start][j] = 0;
        }
        for (int start = j + 1; start < arr.length; start++) {
            arr[i][start] = 0;
        }

        for (int start = i + 1, start2 = j + 1; start < arr.length && start2 < arr.length; start++, start2++) {
            arr[start][start2] = 0;
        }

        for (int start = i - 1, start2 = j + 1; start >= 0 && start2 < arr.length; start++, start2++) {
            arr[start][start2] = 0;
        }

    }

    private int prevIns = -1;

    private boolean insertQueenInCol(int[][] arr, int column) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i][column] == 0 && i != prevIns) {
                prevIns = i;
                insertQueen(arr, i, column);
                return true;
            }
        }
        return false;
    }

    private void removeQueenFromCol(int[][] arr, int column) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i][column] == 1) {
                removeQueen(arr, i, column);
                arr[i][column] = -1;
            }
        }
    }


    private void backtrack(int[][] arr, int row, int column) {
        if (column == arr.length) {
            nQueens++;
            return;
        }

        if (insertQueenInCol(arr, column)) {
            backtrack(arr, row, column + 1);
        } else {
            removeQueenFromCol(arr, column - 1);
            backtrack(arr, row, column - 1);
        }
    }

    public int solveNQueens(int n) {
        int[][] arr = new int[n][n];
        backtrack(arr, 0, 0);
        printMatrix(arr);
//        insertQueen(arr, 1,1);
//        printMatrix(arr);
//        removeQueen(arr, 1,1);
//        printMatrix(arr);
        return nQueens;
    }

    private void swap(char[] arr, int i, int j) {
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public int maximumSwap(int num) {
        int max = num;

        char[] arr = Integer.toString(num).toCharArray();
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[i] < arr[j]) {
                    swap(arr, i, j);
                    max = Math.max(max, Integer.parseInt(new String(arr)));
                    swap(arr, j, i);
                }
            }
        }

        return max;

    }

    @Test
    public void testNQueens() {
        System.out.println(maximumSwap(2736));
        //  System.out.println(solveNQueens(4));
        System.out.println(3 | 5 | 1);

    }

    private int maxOrSubs = 0;

    private int or(List<Integer> lst) {
        return lst.stream().reduce((x, acc) -> x | acc).orElse(0);
    }

    private void combine(int[] nums, int start, int or, List<Integer> curr) {
        if (start >= nums.length) {
            return;
        }
        for (int i = start; i < nums.length; i++) {
            curr.add(nums[i]);
            if (or(curr) == or) {
                maxOrSubs++;
            }
            combine(nums, i + 1, or, curr);
            curr.removeLast();
        }
    }

    public int countMaxOrSubsets(int[] nums) {
        combine(nums, 0, or(Arrays.stream(nums).boxed().collect(Collectors.toList())), new ArrayList<>());
        return maxOrSubs;
    }

    @Test
    public void testMaxCurrSubs() {
        System.out.println(countMaxOrSubsets(arr(3, 2, 1, 5)));
    }


    List<Integer> lst = new ArrayList<>();

    public void preOrder(TreeNode root) {
        if (root == null) {
            return;
        }
        lst.add(root.val);
        preOrder(root.left);
        preOrder(root.right);
    }

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        preOrder(root);
        TreeNode tail = root;
        for (int i = 1; i < lst.size(); i++) {
            tail.right = new TreeNode(lst.get(i));
            tail.left = null;
            tail = tail.right;
        }
    }

    TreeNode prev = null;

    public void flatten2(TreeNode root) {
        if (root == null) {
            return;
        }

        preOrder(root);
        TreeNode tail = root;
        for (int i = 1; i < lst.size(); i++) {
            tail.right = new TreeNode(lst.get(i));
            tail.left = null;
            tail = tail.right;
        }
    }

    @Test
    public void flattenTest() {
        String tree = "3(9(5(6)(2(7)(4)))(10))(1(0)(8))";
        TreeNode root = deserialize(tree);
        TreePrinter.print(root);
        flatten(root);
        TreePrinter.print(root);
    }

    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = new ListNode(0, head);
        ListNode res = prev;
        ListNode curr = head;

        while (curr != null && curr.next != null) {
            prev.next = curr.next;
            ListNode node = curr.next;
            curr.next = curr.next.next;
            node.next = curr;

            prev = curr;
            curr = curr.next;
        }

        return res.next;
    }

    public boolean isHappy(int n) {

        while (n > 9) {

            int sum = 0;
            while (n > 0) {
                int tail = n % 10;
                sum += tail * tail;
                n /= 10;
            }
            n = sum;
        }
        return n == 1;
    }

    @Test
    public void testLinkedSwap() {
        ListNode node = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, null))));
        ListNode.print(node);
        ListNode.print(swapPairs(node));
        System.out.println(isHappy(19));
        System.out.println(isHappy(2));


    }

    public char findKthBit(int n, int k) {
        int[] arr = new int[(int) Math.pow(2, n) - 1];
        int index = 1;
        while (n > 1) {
            arr[index] = 1;
            int r = index + 1;
            for (int l = index - 1; l >= 0; l--) {
                arr[r++] = arr[l] ^ 1;
            }
            index = r;
            n--;

        }
        return (char) ('0' + arr[k - 1]);
    }

    public int reverseBits(int n) {
        int res = 0;
        while (n < 0) {
            int bit = n % 2 == 0 ? 0 : 1;
            res |= bit;
            res <<= 1;
            n >>= 1;
        }

        return res;
    }

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }

        double curr = x;
        for (int i = 2; i <= n; i++) {
            x *= curr;
        }

        return x;
    }

    @Test
    public void checkFindKBit() {
        System.out.println(myPow(2, 10));
        System.out.println(myPow(2, -2));
        System.out.println(myPow(2147483647, -1));
    }

    private int getCell(int[][] dp, int i, int j) {
        try {
            return dp[i][j];
        } catch (IndexOutOfBoundsException e) {
            return 0;
        }
    }

    public int change(int amount, int[] coins) {
        Arrays.sort(coins);
        int m = coins.length, n = amount + 1;
        int[][] dp = new int[m + 1][n];

        for (int i = 0; i < m; i++)
            dp[i][0] = 1;

        for (int i = m - 1; i >= 0; i--)
            for (int capacity = coins[i]; capacity < n; capacity++)
                dp[i][capacity] = dp[i][capacity - coins[i]] + dp[i + 1][capacity];

        printMatrix(dp);
        return dp[0][n - 1];
    }

    @Test
    public void coinChange2() {
        System.out.println(change(3, arr(2)));
        System.out.println(change(5, arr(1, 2, 5)));
        System.out.println(change(100, arr(99, 1)));
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        char[] s1 = text1.toCharArray();
        char[] s2 = text2.toCharArray();

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (s1[i] == s2[j]) {
                    dp[i][j] = 1 + dp[i + 1][j + 1];
                } else {
                    // Else, take the maximum from right or down position
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }

        }

        return dp[0][0];
    }

    @Test
    public void testLongestSubs() {
        System.out.println(longestCommonSubsequence("aec", "abece"));
        System.out.println(longestCommonSubsequence("bsbininm", "jmjkbkjkv"));
    }

    private String operation(Character op, String values) {
        if (op == '!') {
            return values.charAt(0) == 't' ? "f" : "t";
        }
        if (op == '&') {
            for (int i = 0; i < values.length(); i++) {
                if (values.charAt(i) == 'f') {
                    return "f";
                }
            }
        }
        if (op == '|') {
            for (int i = 0; i < values.length(); i++) {
                if (values.charAt(i) == 't') {
                    return "t";
                }
            }
        }
        return values;
    }

    public boolean parseBoolExpr(String expression) {
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            if (c == ',') {
                continue;
            }
            if (c != ')') {
                stack.push(c);
            } else {
                StringBuilder sb = new StringBuilder();
                while (stack.peek() != '(') {
                    sb.insert(0, stack.pop());
                }
                stack.pop();
                char operator = stack.pop();
                String result = operation(operator, sb.toString());
                stack.push(result.charAt(0));
            }
        }
        return stack.pop().toString().equals("t");
    }


    @Test
    public void testEval() {
        System.out.println(parseBoolExpr("|(f,f,f,t)"));
    }

    private int manhattan(int[][] points, int a, int b) {
        return Math.abs(points[a][0] - points[b][0]) + Math.abs(points[a][1] - points[b][1]);
    }

    public int minCostConnectPoints(int[][] points) {
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        minHeap.add(new int[]{0, 0});
        int[] visited = new int[points.length];
        //Arrays.fill(visited, -1);
        int counter = 0;
        //Set<Integer> visited = new HashSet<>();
        int res = 0;

        while (true) {
            int[] minCost = minHeap.remove();
            int currNode = minCost[1];
            if(visited[currNode] > 0) {
                continue;
            }
            res += minCost[0];
            visited[currNode] = 1;
            counter++;
            if (counter >= points.length) {
                break;
            }
            for (int j = currNode; j < points.length; j++) {
                if(visited[j] == 0) {
                    minHeap.add(new int[]{manhattan(points, currNode, j), j});
                }
            }
        }

        return res;
    }

    public List<String> stringSequence(String target) {
        List<String> result = new ArrayList<>();
        String ans = "";
        for(int i = 0; i < target.length(); i++) {
            char curr = 'a';
            char curr1 = target.charAt(i);
            result.add(ans + curr);
            while(curr != curr1) {
                curr += 1;
                result.add(ans + curr);
            }
            ans += curr;
        }
        return result;
    }

    public int getGreatestProperDivisor(int n) {
        if (n % 2 == 0) {
            return n / 2;
        }
        final int sqrtn = (int) Math.sqrt(n);
        for (int i = 3; i <= sqrtn; i += 2) {
            if (n % i == 0) {
                return n / i;
            }
        }
        return 1;
    }

    public boolean isPalindrome(String str) {
        int left = 0, right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }


    public static class TreeNode2 {
        public int val; // Index value
        public char value; // Value associated with the index
        public List<TreeNode2> childs;

        // Default constructor
        public TreeNode2() {
            this.childs = new ArrayList<>();
        }

        // Constructor that takes an index value and associated string value
        public TreeNode2(int val, char value) {
            this.val = val;
            this.value = value;
            this.childs = new ArrayList<>();
        }

        // Method to build the tree from parent indices and values
        public static TreeNode2 buildTree(int[] parent, final String values) {
            // Map to hold nodes by their index
            Map<Integer, TreeNode2> nodeMap = new HashMap<>();
            TreeNode2 root = null;

            for (int i = 0; i < parent.length; i++) {
                // Create or retrieve the current node with the corresponding value
                int finalI = i;
                TreeNode2 currentNode = nodeMap.computeIfAbsent(i, index -> new TreeNode2(index, values.charAt(finalI)));

                // If parent[i] is -1, this is the root node
                if (parent[i] == -1) {
                    root = currentNode;
                } else {
                    // Create or retrieve the parent node
                    int finalI1 = i;
                    TreeNode2 parentNode = nodeMap.computeIfAbsent(parent[i], index -> new TreeNode2(index, values.charAt(finalI1)));
                    // Add the current node as a child of the parent node
                    parentNode.childs.add(currentNode);
                }
            }

            return root;
        }

        // Example of a method to print the tree
        public void printTree(String prefix) {
            System.out.println(prefix + "Node: " + val + ", Value: " + value);
            for (TreeNode2 child : childs) {
                child.printTree(prefix + "  ");
            }
        }
    }

    public boolean[] findAnswer(int[] parent, String s) {
        int n = parent.length;
        TreeNode2 root = TreeNode2.buildTree(parent, s);
        boolean[] result = new boolean[n];
        dfs(root ,"");
        return result;
    }

    private void dfs(TreeNode2 tree, String s) {
       if(tree == null) {
           return;
       }
       for(int i = tree.childs.size() - 1; i >= 0; i--) {
           dfs(tree.childs.get(i), s + tree.value);
       }
        System.out.println(tree.value);
    }

    @Test
    public void testCostConnectPoints() {
        System.out.println(Arrays.toString(findAnswer(arr(-1, 0, 0, 1, 1, 2), "aababa")));

        TreeNode2 root = TreeNode2.buildTree(arr(-1, 0, 0, 1, 1, 2), "aababa");
        root.printTree("a");

    }

    public int numSquares(int n) {
        int[] squares = new int[(int) Math.sqrt(n) + 1];
        for(int i = 0; i < squares.length; i++) {
            squares[i] = i * i;
        }
        int[] dp =  new int[n + 1];
        for(int j = 0; j <= n; j++) {
            dp[j] = j;
        }

        for(int i = 1; i < squares.length; i++) {
            for(int j = squares[i]; j <= n; j++) {
                dp[j] = Math.min(dp[j - squares[i]] + 1, dp[j]);
            }
        }
        //    0   1   2   3   1   2   3   4   2   1   2   3   3
        System.out.println(Arrays.toString(dp));
        return dp[n];
    }

    @Test
    public void testNumSqueares() {
        System.out.println(numSquares(12));
    }

    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, Comparator.comparingInt(o -> o[0]));
        int n = pairs.length;
        int[] dp = new int[n];
        dp[n - 1] = 1;

        for(int i = n - 2; i >= 0; i--) {
            dp[i] = dp[i + 1];
            for(int j = i + 1; j < n; j++) {
                if(pairs[i][1] < pairs[j][0]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }


        return dp[0];
    }

    @Test
    public void testFindLongestChain() {
        System.out.println(findLongestChain(matrix("[[1,2],[7,8],[4,5]]")));
        System.out.println(findLongestChain(matrix("[[1,2],[2,3],[3,4]]")));
    }

    int ssize = 0;

    private int dfs(int[][] arr, int[][] dp, int i, int j, int prev) {
        if(i < 0 || i >= arr.length || j < 0 || j >= arr[0].length || arr[i][j] <= prev ) {
            return 0;
        }
        if(dp[i][j] != 0) {
            return dp[i][j];
        }
        int curr = arr[i][j];
        arr[i][j] = -1;
        int up = dfs(arr, dp, i + 1, j, curr);
        int b = dfs(arr, dp, i, j + 1, curr);
        int l = dfs(arr, dp, i - 1, j, curr);
        int r = dfs(arr, dp, i, j - 1, curr);
        arr[i][j] = curr;
        int res = Math.max(Math.max(Math.max(l, r), b), up) + 1;
        dp[i][j] = res;
        return res;
    }

    public int longestIncreasingPath(int[][] matrix) {
        printMatrix(matrix);
        int[][] dp = new int[matrix.length][matrix[0].length];
        int max = 0;
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[0].length; j++) {
                if(dp[i][j] == 0) {
                    dfs(matrix, dp, i, j, -1);
                }
                max = Math.max(dp[i][j], max);
            }
        }
        printMatrix(dp);
        return max;
    }

    @Test
    public void testlongestIncreasingPath() {
        System.out.println(longestIncreasingPath(matrix("[[1]]")));
        System.out.println(longestIncreasingPath(matrix("[[1,2],[2,3]]")));
        System.out.println(longestIncreasingPath(matrix("[[9,9,4],[6,6,8],[2,1,1]]")));
        System.out.println(longestIncreasingPath(matrix("[[3,4,5],[3,2,6],[2,2,1]]")));
        System.out.println(longestIncreasingPath(matrix("[[7,7,5],[2,4,6],[8,2,0]]")));
    }



    int maxSize = 0;
    private void backtrack(char[] s, int start, Set<String> set) {
        if (set.size() + (s.length - start) <= maxSize) {
            return; // Early exit, no point in exploring further
        }
        if(start == s.length) {
            maxSize = Math.max(maxSize, set.size());
            System.out.println(set);
            return;
        }

        for(int i = start; i < s.length; i++) {
            String curr = new String(Arrays.copyOfRange(s, start, i + 1));
            if(!set.contains(curr)) {
                set.add(curr);
                backtrack(s, i + 1, set);
                set.remove(curr);
            }
        }
    }

    public int maxUniqueSplit(String s) {
        backtrack(s.toCharArray(), 0, new HashSet<>());
        return maxSize;
    }

    @Test
    public void testMaxUniqueSplit() {
        System.out.println(maxUniqueSplit("aabca"));
    }
}
