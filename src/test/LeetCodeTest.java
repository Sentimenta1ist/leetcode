package test;

import org.junit.jupiter.api.Test;
import src.structures.DoubleLinkedList;
import src.structures.ListNode;
import src.structures.TreeNode;
import src.structures.TreePrinter;
import src.structures.Trie;

import java.util.Arrays;
import java.util.List;

import static src.LeetCode.*;


public class LeetCodeTest {

    @Test
    public void testMerge() {
        int[] nums1 = new int[]{7, 8, 0, 0, 0};
        int[] nums2 = new int[]{1, 3, 6};

        merge(nums1, 2, nums2, 3);
        System.out.println(Arrays.toString(Arrays.stream(nums1).toArray()));
    }

    @Test
    public void testMerge2() {
        int[] nums1 = new int[]{1, 2, 3, 0, 0, 0};
        int[] nums2 = new int[]{2, 5, 6};

        merge(nums1, 3, nums2, 3);
        System.out.println(Arrays.toString(Arrays.stream(nums1).toArray()));
    }

    @Test
    public void testRemoveElement() {
        int[] nums1 = new int[]{0, 1, 2, 2, 3, 0, 4, 2};
        int num = 2;

        removeElement(nums1, num);
        System.out.println(Arrays.toString(Arrays.stream(nums1).toArray()));

        int[] nums2 = new int[]{3, 2, 2, 3};
        int num2 = 3;

        removeElement(nums2, num2);
        System.out.println(Arrays.toString(Arrays.stream(nums2).toArray()));
    }

    @Test
    public void testRemoveDuplicates() {
        int[] nums = new int[]{0, 1, 1, 1, 1, 2, 2, 3, 3, 4};

        removeDuplicates(nums);
        System.out.println(Arrays.toString(Arrays.stream(nums).toArray()));

    }

    @Test
    public void testMajorityElement() {
        int[] nums = new int[]{2, 2, 1, 1, 1, 2, 2};

        System.out.println(majorityElement(nums));
    }

    @Test
    public void testRotateArray() {
        int[] nums = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        rotateArray(nums, 4);
        System.out.println(Arrays.toString(nums));
    }

    @Test
    public void testPartition() {
        ListNode node2 = new ListNode(3, new ListNode(8, new ListNode(5, new ListNode(1, new ListNode(2, null)))));
        ListNode node1 = new ListNode(2, new ListNode(1, null));
        ListNode.print(node2);
        partition(node2, 5);
        ListNode.print(node2);

        ListNode.print(node1);
        ListNode.print(partition(node1, 2));

        ListNode.print(partition(null, 2));
    }

    @Test
    public void testReverse() {
        DoubleLinkedList myDLL = new DoubleLinkedList(1);
        myDLL.append(2);
        myDLL.append(3);
        myDLL.append(4);
        myDLL.append(5);

        System.out.println("DLL before reverse:");
        myDLL.printList();

        myDLL.reverse();

        System.out.println("\nDLL after reverse:");
        myDLL.printList();

    }

    @Test
    public void isSubPathTests() {
        // Manually creating the tree from the BFS order
        TreeNode root = new TreeNode(1); // Level 0

        // Level 1
        root.left = new TreeNode(4);
        root.right = new TreeNode(4);

        // Level 2
        root.left.left = null; // Null as per the input array
        root.left.right = new TreeNode(2);
        root.right.left = new TreeNode(2);
        root.right.right = null; // Null as per the input array

        // Level 3
        root.left.right.left = new TreeNode(1);
        root.left.right.right = null; // Null as per the input array
        root.right.left.left = new TreeNode(6);
        root.right.left.right = new TreeNode(8);

        // Level 4
        root.left.right.left.left = null; // Null as per the input array
        root.left.right.left.right = null; // Null as per the input array
        root.right.left.left.left = null; // Null as per the input array
        root.right.left.left.right = null; // Null as per the input array
        root.right.left.right.left = new TreeNode(1);
        root.right.left.right.right = new TreeNode(3);

        ListNode node = new ListNode(4, new ListNode(2, new ListNode(8, null)));

        System.out.println(isSubPath(node, root));
    }

    @Test
    public void testModifiedList() {
        int[] nums = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        modifiedList(nums, null);
    }


    @Test
    public void testSumRange() {
        assert sumRange(new int[]{-2, 0, 3, -5, 2, -1}, 0, 2) == 1;
        assert sumRange(new int[]{-2, 0, 3, -5, 2, -1}, 2, 5) == -1;
        assert sumRange(new int[]{-2, 0, 3, -5, 2, -1}, 0, 5) == -3;
    }

    @Test
    public void testFindMaxLength() {
        assert findMaxLength(new int[]{0, 1, 1, 1, 0, 0}) == 6;
        assert findMaxLength(new int[]{0, 1, 1, 1, 1, 1, 0, 0}) == 4;
        assert findMaxLength(new int[]{1, 1, 0, 0, 1, 1, 0, 1, 0, 1}) == 8;
        assert findMaxLength(new int[]{1, 0, 1, 1, 0, 1, 1, 1, 0, 1}) == 4;
        assert findMaxLength(new int[]{1, 0, 0, 0, 1, 1, 0, 1, 1, 0}) == 10;
    }

    @Test
    public void testBinaryRotated() {
        System.out.println(findLeft(new int[]{1, 3}));
        System.out.println(searchInShifted(new int[]{3, 1}, 3));
    }

    @Test
    public void testBinarySearch() {
        System.out.println(binarySearch(new int[]{1, 2, 3, 4, 5, 6, 7}, 7));
        System.out.println(binarySearchRec(new int[]{1, 2, 3, 4, 5, 6, 7}, 7, 0, 7));
    }

    @Test
    public void testFindDuplicate() {
        assert findDuplicate(new int[]{3, 1, 3, 4, 2}) == 3;
        assert findDuplicate(new int[]{1, 3, 4, 2, 2}) == 2;
        assert findDuplicate(new int[]{3, 3, 3, 3, 3}) == 3;
    }

    @Test
    public void testPeakIndex() {
        assert peakIndexInMountainArray(new int[]{3, 5, 3, 2, 0}) == 1;
    }

    @Test
    public void testTwoSum2() {
        assert Arrays.equals(twoSum2(new int[]{-1, -1, 1, 1}, -2), new int[]{0, 1});
        assert Arrays.equals(twoSum2(new int[]{-1000, -1, 0, 1}, 1), new int[]{2, 3});
        assert Arrays.equals(twoSum2(new int[]{2, 7, 11, 15}, 9), new int[]{0, 1});
    }

    @Test
    public void testTwoSum() {
        assert Arrays.equals(twoSum(new int[]{2, 7, 11, 15}, 9), new int[]{1, 0});
    }

    @Test
    public void testFindMinRotated() {
        //System.out.println(findMin(new int[]{3, 4, 5, 1, 2}));
        assert findMin(new int[]{3, 1, 2}) == 1;
        assert findMin(new int[]{3, 4, 5, 1, 2}) == 1;
        assert findMin(new int[]{4, 5, 6, 7, 0, 1, 2}) == 0;
        assert findMin(new int[]{5, 1, 2, 3, 4}) == 1;
    }

    @Test
    public void testSetMatrixZeros() {
        int[][] arr4 = new int[][]{{0, 0, 0, 5}, {4, 3, 1, 4}, {0, 1, 1, 4}, {1, 2, 1, 3}, {0, 0, 1, 1}};
        printMatrix(arr4);
        setZeroes(arr4);
        printMatrix(arr4);

        int[][] arr = new int[][]{{1, 2, 3, 4}, {5, 0, 7, 8}, {0, 10, 11, 12}, {13, 14, 15, 0}};
        printMatrix(arr);
        setZeroes(arr);
        printMatrix(arr);

        int[][] arr2 = new int[][]{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
        printMatrix(arr2);
        setZeroes(arr2);
        printMatrix(arr2);

        int[][] arr3 = new int[][]{{1, 0}};
        printMatrix(arr3);
        setZeroes(arr3);
        printMatrix(arr3);
    }

    @Test
    public void testMaxSlidingWindow() {
        assert Arrays.equals(maxSlidingWindow(new int[]{9, 8, 9, 8}, 3), new int[]{9, 9});
        assert Arrays.equals(maxSlidingWindow(new int[]{-6, -10, -7, -1, -9, 9, -8, -4, 10, -5, 2, 9, 0, -7, 7, 4, -2, -10, 8, 7}, 7), new int[]{9, 9, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 8, 8});
        assert Arrays.equals(maxSlidingWindow(new int[]{-6, 10, -5, 2, 9, 0, -7, 7, 4}, 7), new int[]{10, 10, 9});
        assert Arrays.equals(maxSlidingWindow(new int[]{9, 10, 9, -7, -4, -8, 2, -6}, 5), new int[]{10, 10, 9, 2});
        assert Arrays.equals(maxSlidingWindow(new int[]{1, 3, 1, 2, 0, 5}, 3), new int[]{3, 3, 2, 5});
        assert Arrays.equals(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3), new int[]{3, 3, 5, 5, 6, 7});
        assert Arrays.equals(maxSlidingWindow(new int[]{1}, 1), new int[]{1});
        assert Arrays.equals(maxSlidingWindow(new int[]{1, -1}, 1), new int[]{1, -1});
    }

    @Test
    public void testProductExceptSelf() {
        assert Arrays.equals(productExceptSelf(new int[]{1, 2, 3, 4}), new int[]{24, 12, 8, 6});
        assert Arrays.equals(productExceptSelf(new int[]{-1, 1, 0, -3, 3}), new int[]{0, 0, 9, 0, 0});
    }

    @Test
    public void testSubarraySum() {
        assert subarraySum(new int[]{1, 1, 1}, 1) == 3;
        assert subarraySum(new int[]{1, 2, 3, 4}, 3) == 2;
    }

    @Test
    public void testPalindromeList() {
        ListNode node = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(2, new ListNode(1, null)))));
        ListNode.print(node);
        System.out.println(isPalindrome(node));
    }

    @Test
    public void testRemoveElementLinkedList() {
        ListNode node = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(1, new ListNode(1, null)))));
        ListNode.print(node);
        removeElements(node, 1);
        ListNode.print(node);
    }

    @Test
    public void testRob() {
        System.out.println(rob(new int[]{4, 1, 2, 7, 5, 3, 1}));
        System.out.println(rob(new int[]{1, 3, 1, 3, 100}));
    }

    @Test
    public void testMaxSubArray() {
        System.out.println(maxSubArray(new int[]{-2, -1}));
        System.out.print(1);
        System.out.println(maxSubArray(new int[]{5, 4, -1, 7, 8}));
        System.out.print(23);
        System.out.println(maxSubArray(new int[]{-2, 1}));
        System.out.print(1);
    }

    @Test
    public void testCoinChange() {
        System.out.println(coinChange(new int[]{1, 2, 5}, 11));
        System.out.println(coinChange(new int[]{3}, 11));
    }

    @Test
    public void testBinaryTreePaths() {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(5);
        root.left.right = null;

        List<String> res = binaryTreePaths(root);
        System.out.println(res);
        TreePrinter.print(root);
    }

    @Test
    public void testWordExist() {
        char[][] arr = new char[][]{
                {'A', 'B', 'C', 'E'},
                {'S', 'F', 'C', 'S'},
                {'A', 'D', 'E', 'E'}};

        System.out.println(exist(arr, "A1"));
        System.out.println(Arrays.deepToString(arr));
    }

    @Test
    public void testTrie() {
        Trie trie = new Trie();
        trie.insert("apple");
        trie.insert("app");
        trie.search("app");     // return True
        trie.search("apple");   // return True
        trie.search("app");     // return False
        trie.startsWith("app"); // return True
        Trie trie1 = new Trie();
        System.out.println(trie1.startsWith("appl"));
        System.out.println(trie1.search("appl"));
        trie1.insert("apple");
        System.out.println(trie1.startsWith("appl"));
        System.out.println(trie1.search("appl"));
        trie1.insert("appra");

        trie1.insert("appda");
        System.out.println(trie1.search("appda"));
        System.out.println(trie1.search("appra"));
        trie1.print();
        System.out.println("-----");

    }
}
