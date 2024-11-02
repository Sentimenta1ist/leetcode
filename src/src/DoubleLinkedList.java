package src;

public class DoubleLinkedList {

    private DLNode head;
    private DLNode tail;
    private int length;

    static class DLNode {
        int value;
        DLNode next;
        DLNode prev;

        DLNode(int value) {
            this.value = value;
        }
    }

    public DoubleLinkedList(int value) {
        DLNode newDLNode = new DLNode(value);
        head = newDLNode;
        tail = newDLNode;
        length = 1;
    }

    public DLNode getHead() {
        return head;
    }

    public DLNode getTail() {
        return tail;
    }

    public int getLength() {
        return length;
    }

    public void printList() {
        DLNode temp = head;
        while (temp != null) {
            System.out.println(temp.value);
            temp = temp.next;
        }
    }

    public void printAll() {
        if (length == 0) {
            System.out.println("Head: null");
            System.out.println("Tail: null");
        } else {
            System.out.println("Head: " + head.value);
            System.out.println("Tail: " + tail.value);
        }
        System.out.println("Length:" + length);
        System.out.println("\nDoubly Linked List:");
        if (length == 0) {
            System.out.println("empty");
        } else {
            printList();
        }
    }

    public void makeEmpty() {
        head = null;
        tail = null;
        length = 0;
    }

    public void append(int value) {
        DLNode newDLNode = new DLNode(value);
        if (length == 0) {
            head = newDLNode;
            tail = newDLNode;
        } else {
            tail.next = newDLNode;
            newDLNode.prev = tail;
            tail = newDLNode;
        }
        length++;
    }

    public DLNode get(int index) {
        if (index < 0 || index >= length) return null;
        DLNode temp = head;
        if (index < length / 2) {
            for (int i = 0; i < index; i++) {
                temp = temp.next;
            }
        } else {
            temp = tail;
            for (int i = length - 1; i > index; i--) {
                temp = temp.prev;
            }
        }
        return temp;
    }

    public void reverse() {
        DLNode temp = head;
        while (temp != null) {
            DLNode DLNode = new DLNode(0);
            DLNode.next = temp.next;
            temp.next = temp.prev;
            temp.prev = DLNode.next;
            temp = temp.prev;
        }

        if (head != null && tail != null) {
            DLNode DLNode = new DLNode(head.value);
            DLNode.prev = head.prev;
            head = tail;
            tail = DLNode;
        }
    }

}