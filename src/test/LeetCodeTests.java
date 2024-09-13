package test;

import org.junit.jupiter.api.Test;
import src.structures.DoubleLinkedList;
import src.structures.ListNode;
import src.structures.TreeNode;

import java.util.Arrays;

import static src.LeetCode.isSubPath;
import static src.LeetCode.majorityElement;
import static src.LeetCode.merge;
import static src.LeetCode.modifiedList;
import static src.LeetCode.partition;
import static src.LeetCode.removeDuplicates;
import static src.LeetCode.removeElement;
import static src.LeetCode.rotateArray;
import static src.LeetCodeTasks.sumRange;


public class LeetCodeTests {

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
}
