package src;

import org.junit.jupiter.api.Test;
import src.structures.ListNode;
import src.structures.TreeNode;
import src.structures.TreePrinter;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Queue;

public class LeetCode {

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
        if(root == null) {
            return false;
        }
        if(root.left == null && root.right == null) {
            return targetSum - root.val == 0;
        } else{
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
}

