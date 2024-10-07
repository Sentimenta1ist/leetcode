package src.structures;

public class ListNode {
    public int val;
    public ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }

    public static void print(ListNode head) {
        ListNode pointer = head;
        while(pointer != null) {
            System.out.printf("%d->", pointer.val);
            pointer = pointer.next;
        }
        System.out.printf("null\n");


    }
}