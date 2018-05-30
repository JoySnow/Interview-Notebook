
* [数据结构相关](#数据结构相关)
    * [栈和队列](#栈和队列)
    * [哈希表](#哈希表)
    * [字符串](#字符串)
    * [数组与矩阵](#数组与矩阵)
        * [1-n 分布](#1-n-分布)
        * [有序矩阵](#有序矩阵)
    * [链表](#链表)
    * [树](#树)
        * [递归](#递归)
        * [层次遍历](#层次遍历)
        * [前中后序遍历](#前中后序遍历)
        * [BST](#bst)
        * [Trie](#trie)
    * [图](#图)
    * [位运算](#位运算)



# 数据结构相关

## 栈和队列

**用栈实现队列**

[Leetcode : 232. Implement Queue using Stacks (Easy)](https://leetcode.com/problems/implement-queue-using-stacks/description/)

~~~
Implement the following operations of a queue using stacks.

Note:
- You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
- You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).
~~~

List 实现：(not follow the description)
https://docs.python.org/2/tutorial/datastructures.html#using-lists-as-stacks

```python
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.q.append(x)        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.q.pop(0)        

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.q[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self.q) < 1
```

两个栈实现：

```python
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        https://docs.python.org/2/tutorial/datastructures.html#using-lists-as-stacks
        """
        self.s1 = []  # maintain the stack top(index 0 for bottom) as queue's top,
        self.s2 = []  # help with the covert for each insert

    def push(self, x):
        """
        O(n) Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        while self.s1:
            self.s2.append(self.s1.pop())
        self.s1.append(x)
        while self.s2:
            self.s1.append(self.s2.pop())

    def pop(self):
        """
        O(1) Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.s1.pop()

    def peek(self):
        """
        O(1) Get the front element.
        :rtype: int
        """
        return self.s1[-1]        

    def empty(self):
        """
        O(1) Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self.s1) < 1        

```

**用队列实现栈**

[Leetcode : 225. Implement Stack using Queues (Easy)](https://leetcode.com/problems/implement-stack-using-queues/description/)

```python
class MyStack(object):
    def __init__(self):
        """
        Use one queue only.
        Push: O(N), use the queue FIFO, pop and repush it to maintain the first
              element is the statck's top.
        Pop: O(1)
        https://docs.python.org/2/tutorial/datastructures.html#using-lists-as-queues
        The dequeue.popleft() can be pop the itme at [0].
        """
        from collections import deque
        self.q1 = deque([])

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.q1.append(x)
        for i in xrange(len(self.q1)-1):
            self.q1.append(self.q1.popleft())

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.q1.popleft()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.q1[0]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return len(self.q1) < 1
```

**最小值栈**

[Leetcode : 155. Min Stack (Easy)](https://leetcode.com/problems/min-stack/description/)

用两个栈实现，一个存储数据，一个存储最小值。

```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.s = []   #存储数据
        self.f = []   #存储最小值

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.s.append(x)
        if not self.f:
            self.f.append(x)
        else:
            minx = self.getMin()
            self.f.append(minx if minx < x else x)  

    def pop(self):
        """
        :rtype: void
        """
        self.s.pop()
        self.f.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.s[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.f[-1]
```

对于实现最小值队列问题，可以先将队列使用栈来实现，然后就将问题转换为最小值栈，这个问题出现在 编程之美：3.7。

**用栈实现括号匹配**

[Leetcode : 20. Valid Parentheses (Easy)](https://leetcode.com/problems/valid-parentheses/description/)

```html
Input:  "()[]{}", "([)]", "("

Output : true, false, false
```

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        st = []
        left = ('[', '{', '(')
        right = (']', '}', ')')
        mapp = dict(zip(right, left))

        for v in s:
            if v in left:
                st.append(v)
            elif v in right and len(st) > 0 and mapp[v] == st[-1]:
                st.pop()
            else:
                return False

        return True if len(st) == 0 else False
```

**数组中元素与下一个比它大的元素之间的距离**

[Leetcode : 739. Daily Temperatures (Medium)](https://leetcode.com/problems/daily-temperatures/description/)

```html
Input: [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```


在遍历数组时用 Stack 把数组中的数存起来，如果当前遍历的数比栈顶元素来的大，说明栈顶元素的下一个比它大的数就是当前元素。

```python
public int[] dailyTemperatures(int[] temperatures) {
    int n = temperatures.length;
    int[] ret = new int[n];
    Stack<Integer> stack = new Stack<>();
    for(int i = 0; i < n; i++) {
        while(!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
            int idx = stack.pop();
            ret[idx] = i - idx;
        }
        stack.add(i);
    }
    return ret;
}
```

**在另一个数组中比当前元素大的下一个元素**

[Leetcode : 496. Next Greater Element I (Easy)](https://leetcode.com/problems/next-greater-element-i/description/)

```html
Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
Output: [-1,3,-1]
```

```python
public int[] nextGreaterElement(int[] nums1, int[] nums2) {
    Map<Integer, Integer> map = new HashMap<>();
    Stack<Integer> stack = new Stack<>();
    for(int num : nums2){
        while(!stack.isEmpty() && num > stack.peek()){
            map.put(stack.pop(), num);
        }
        stack.add(num);
    }
    int[] ret = new int[nums1.length];
    for(int i = 0; i < nums1.length; i++){
        if(map.containsKey(nums1[i])) ret[i] = map.get(nums1[i]);
        else ret[i] = -1;
    }
    return ret;
}
```

**循环数组中比当前元素大的下一个元素**

[Leetcode : 503. Next Greater Element II (Medium)](https://leetcode.com/problems/next-greater-element-ii/description/)

```python
public int[] nextGreaterElements(int[] nums) {
    int n = nums.length, next[] = new int[n];
    Arrays.fill(next, -1);
    Stack<Integer> stack = new Stack<>();
    for (int i = 0; i < n * 2; i++) {
        int num = nums[i % n];
        while (!stack.isEmpty() && nums[stack.peek()] < num)
            next[stack.pop()] = num;
        if (i < n) stack.push(i);
    }
    return next;
}
```

## 哈希表

利用 Hash Table 可以快速查找一个元素是否存在等问题，但是需要一定的空间来存储。在优先考虑时间复杂度的情况下，可以利用 Hash Table 这种空间换取时间的做法。

Java 中的  **HashSet**  用于存储一个集合，并以 O(1) 的时间复杂度查找元素是否在集合中。

如果元素有穷，并且范围不大，那么可以用一个布尔数组来存储一个元素是否存在，例如对于只有小写字符的元素，就可以用一个长度为 26 的布尔数组来存储一个字符集合，使得空间复杂度降低为 O(1)。

Java 中的  **HashMap**  主要用于映射关系，从而把两个元素联系起来。

在对一个内容进行压缩或者其它转换时，利用 HashMap 可以把原始内容和转换后的内容联系起来。例如在一个简化 url 的系统中（[Leetcdoe : 535. Encode and Decode TinyURL (Medium)](https://leetcode.com/problems/encode-and-decode-tinyurl/description/)），利用 HashMap 就可以存储精简后的 url 到原始 url 的映射，使得不仅可以显示简化的 url，也可以根据简化的 url 得到原始 url 从而定位到正确的资源。

HashMap 也可以用来对元素进行计数统计，此时键为元素，值为计数。和 HashSet 类似，如果元素有穷并且范围不大，可以用整型数组来进行统计。

**数组中的两个数和为给定值**

[Leetcode : 1. Two Sum (Easy)](https://leetcode.com/problems/two-sum/description/)

可以先对数组进行排序，然后使用双指针方法或者二分查找方法。这样做的时间复杂度为 O(nlg<sub>n</sub>)，空间复杂度为 O(1)。

用 HashMap 存储数组元素和索引的映射，在访问到 nums[i] 时，判断 HashMap 中是否存在 target - nums[i] ，如果存在说明 target - nums[i] 所在的索引和 i 就是要找的两个数。该方法的时间复杂度为 O(n)，空间复杂度为 O(n)，使用空间来换取时间。

```python
public int[] twoSum(int[] nums, int target) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for(int i = 0; i < nums.length; i++){
        if(map.containsKey(target - nums[i])) return new int[]{map.get(target - nums[i]), i};
        else map.put(nums[i], i);
    }
    return null;
}
```

**最长和谐序列**

[Leetcode : 594. Longest Harmonious Subsequence (Easy)](https://leetcode.com/problems/longest-harmonious-subsequence/description/)

```html
Input: [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].
```

和谐序列中最大数和最小数只差正好为 1。

```python
public int findLHS(int[] nums) {
    Map<Long, Integer> map = new HashMap<>();
    for (long num : nums) {
        map.put(num, map.getOrDefault(num, 0) + 1);
    }
    int result = 0;
    for (long key : map.keySet()) {
        if (map.containsKey(key + 1)) {
            result = Math.max(result, map.get(key + 1) + map.get(key));
        }
    }
    return result;
}
```

## 字符串

**两个字符串包含的字符是否完全相同**

[Leetcode : 242. Valid Anagram (Easy)](https://leetcode.com/problems/valid-anagram/description/)

```html
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.
```

字符串只包含小写字符，总共有 26 个小写字符。可以用 Hash Table 来映射字符与出现次数，因为键值范围很小，因此可以使用长度为 26 的整型数组对字符串出现的字符进行统计，比较两个字符串出现的字符数量是否相同。

```python
public boolean isAnagram(String s, String t) {
    int[] cnts = new int[26];
    for(int i  = 0; i < s.length(); i++) cnts[s.charAt(i) - 'a'] ++;
    for(int i  = 0; i < t.length(); i++) cnts[t.charAt(i) - 'a'] --;
    for(int i  = 0; i < 26; i++) if(cnts[i] != 0) return false;
    return true;
}
```

**字符串同构**

[Leetcode : 205. Isomorphic Strings (Easy)](https://leetcode.com/problems/isomorphic-strings/description/)

```html
Given "egg", "add", return true.
Given "foo", "bar", return false.
Given "paper", "title", return true.
```

记录一个字符上次出现的位置，如果两个字符串中某个字符上次出现的位置一样，那么就属于同构。

```python
public boolean isIsomorphic(String s, String t) {
    int[] m1 = new int[256];
    int[] m2 = new int[256];
    for(int i = 0; i < s.length(); i++){
        if(m1[s.charAt(i)] != m2[t.charAt(i)]) {
            return false;
        }
        m1[s.charAt(i)] = i + 1;
        m2[t.charAt(i)] = i + 1;
    }
    return true;
}
```

**计算一组字符集合可以组成的回文字符串的最大长度**

[Leetcode : 409. Longest Palindrome (Easy)](https://leetcode.com/problems/longest-palindrome/description/)

```html
Input : "abccccdd"
Output : 7
Explanation : One longest palindrome that can be built is "dccaccd", whose length is 7.
```

使用长度为 128 的整型数组来统计每个字符出现的个数，每个字符有偶数个可以用来构成回文字符串。因为回文字符串最中间的那个字符可以单独出现，所以如果有单独的字符就把它放到最中间。

```python
public int longestPalindrome(String s) {
    int[] cnts = new int[128]; // ascii 码总共 128 个
    for(char c : s.toCharArray()) cnts[c]++;
    int ret = 0;
    for(int cnt : cnts)  ret += (cnt / 2) * 2;
    if(ret < s.length()) ret++; // 这个条件下 s 中一定有单个未使用的字符存在，可以把这个字符放到回文的最中间
    return ret;
}
```

**判断一个整数是否是回文数**

[Leetcode : 9. Palindrome Number (Easy)](https://leetcode.com/problems/palindrome-number/description/)

要求不能使用额外空间，也就不能将整数转换为字符串进行判断。

将整数分成左右两部分，右边那部分需要转置，然后判断这两部分是否相等。

```python
public boolean isPalindrome(int x) {
    if(x == 0) return true;
    if(x < 0) return false;
    if(x % 10 == 0) return false;
    int right = 0;
    while(x > right){
        right = right * 10 + x % 10;
        x /= 10;
    }
    return x == right || x == right / 10;
}
```

**回文子字符串**

[Leetcode : 647. Palindromic Substrings (Medium)](https://leetcode.com/problems/palindromic-substrings/description/)

```html
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
```

解决方案是从字符串的某一位开始，尝试着去扩展子字符串。

```python
private int cnt = 0;
public int countSubstrings(String s) {
    for(int i = 0; i < s.length(); i++) {
        extendSubstrings(s, i, i);    // 奇数长度
        extendSubstrings(s, i, i + 1); // 偶数长度
    }
    return cnt;
}

private void extendSubstrings(String s, int start, int end) {
    while(start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
        start--;
        end++;
        cnt++;
    }
}
```

**统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数**

[Leetcode : 696. Count Binary Substrings (Easy)](https://leetcode.com/problems/count-binary-substrings/description/)

```html
Input: "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
```

```python
public int countBinarySubstrings(String s) {
    int preLen = 0, curLen = 1, ret = 0;
    for(int i = 1; i < s.length(); i++){
        if(s.charAt(i) == s.charAt(i-1)) curLen++;
        else{
            preLen = curLen;
            curLen = 1;
        }

        if(preLen >= curLen) ret++;
    }
    return ret;
}
```

**字符串循环移位包含**

[ 编程之美：3.1](#)

```html
s1 = AABCD, s2 = CDAA
Return : true
```

给定两个字符串 s1 和 s2 ，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。

s1 进行循环移位的结果是 s1s1 的子字符串，因此只要判断 s2 是否是 s1s1 的子字符串即可。

**字符串循环移位**

[ 编程之美：2.17](#)

将字符串向右循环移动 k 位。

例如 abcd123 向右移动 3 位 得到 123abcd

将 abcd123 中的 abcd 和 123 单独逆序，得到 dcba321，然后对整个字符串进行逆序，得到 123abcd。

**字符串中单词的翻转**

[程序员代码面试指南](#)

例如将 "I am a student" 翻转成 "student a am I"

将每个单词逆序，然后将整个字符串逆序。

## 数组与矩阵

**把数组中的 0 移到末尾**

[Leetcode : 283. Move Zeroes (Easy)](https://leetcode.com/problems/move-zeroes/description/)

```html
For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
```

```python
public void moveZeroes(int[] nums) {
    int n = nums.length;
    int idx = 0;
    for(int i = 0; i < n; i++){
        if(nums[i] != 0) nums[idx++] = nums[i];
    }
    while(idx < n){
        nums[idx++] = 0;
    }
}
```

### 1-n 分布

**一个数组元素在 [1, n] 之间，其中一个数被替换为另一个数，找出丢失的数和重复的数**

[Leetcode : 645. Set Mismatch (Easy)](https://leetcode.com/problems/set-mismatch/description/)

```html
Input: nums = [1,2,2,4]
Output: [2,3]
```

最直接的方法是先对数组进行排序，这种方法时间复杂度为 O(nlog<sub>n</sub>).本题可以以 O(n) 的时间复杂度、O(1) 空间复杂度来求解。

主要思想是让通过交换数组元素，使得数组上的元素在正确的位置上

遍历数组，如果第 i 位上的元素不是 i + 1 ，那么就交换第 i 位 和 nums[i] - 1 位上的元素，使得 num[i] - 1 的元素为 nums[i] ，也就是该位的元素是正确的。交换操作需要循环进行，因为一次交换没办法使得第 i 位上的元素是正确的。但是要交换的两个元素可能就是重复元素，那么循环就可能永远进行下去，终止循环的方法是加上 nums[i] != nums[nums[i] - 1 条件。

类似题目：

- [Leetcode :448. Find All Numbers Disappeared in an Array (Easy)](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/)，寻找所有丢失的元素
- [Leetcode : 442. Find All Duplicates in an Array (Medium)](https://leetcode.com/problems/find-all-duplicates-in-an-array/description/)，寻找所有重复的元素。

```python
public int[] findErrorNums(int[] nums) {
    for(int i = 0; i < nums.length; i++){
        while(nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]) {
            swap(nums, i, nums[i] - 1);
        }
    }

    for(int i = 0; i < nums.length; i++){
        if(i + 1 != nums[i]) {
            return new int[]{nums[i], i + 1};
        }
    }

    return null;
}

private void swap(int[] nums, int i, int j){
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}
```

**找出数组中重复的数，数组值在 [0, n-1] 之间**

[Leetcode : 287. Find the Duplicate Number (Medium)](https://leetcode.com/problems/find-the-duplicate-number/description/)

二分查找解法：

```python
public int findDuplicate(int[] nums) {
     int l = 1, h = nums.length - 1;
     while (l <= h) {
         int mid = l + (h - l) / 2;
         int cnt = 0;
         for (int i = 0; i < nums.length; i++) {
             if (nums[i] <= mid) cnt++;
         }
         if (cnt > mid) h = mid - 1;
         else l = mid + 1;
     }
     return l;
}
```

双指针解法，类似于有环链表中找出环的入口：

```python
public int findDuplicate(int[] nums) {
      int slow = nums[0], fast = nums[nums[0]];
      while (slow != fast) {
          slow = nums[slow];
          fast = nums[nums[fast]];
      }

      fast = 0;
      while (slow != fast) {
          slow = nums[slow];
          fast = nums[fast];
      }
      return slow;
}
```

### 有序矩阵

有序矩阵指的是行和列分别有序的矩阵。一般可以利用有序性使用二分查找方法。

```html
[
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
]
```

**有序矩阵查找**

[Leetocde : 240. Search a 2D Matrix II (Medium)](https://leetcode.com/problems/search-a-2d-matrix-ii/description/)

```python
public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
    int m = matrix.length, n = matrix[0].length;
    int row = 0, col = n - 1;
    while (row < m && col >= 0) {
        if (target == matrix[row][col]) return true;
        else if (target < matrix[row][col]) col--;
        else row++;
    }
    return false;
}
```

**有序矩阵的 Kth Element**

[Leetcode : 378. Kth Smallest Element in a Sorted Matrix ((Medium))](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/)

```html
matrix = [
  [ 1,  5,  9],
  [10, 11, 13],
  [12, 13, 15]
],
k = 8,

return 13.
```

解题参考：[Share my thoughts and Clean Java Code](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85173)

二分查找解法：

```python
public int kthSmallest(int[][] matrix, int k) {
    int m = matrix.length, n = matrix[0].length;
    int lo = matrix[0][0], hi = matrix[m - 1][n - 1];
    while(lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cnt = 0;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n && matrix[i][j] <= mid; j++) {
                cnt++;
            }
        }
        if(cnt < k) lo = mid + 1;
        else hi = mid - 1;
    }
    return lo;
}
```

堆解法：

```python
public int kthSmallest(int[][] matrix, int k) {
    int m = matrix.length, n = matrix[0].length;
    PriorityQueue<Tuple> pq = new PriorityQueue<Tuple>();
    for(int j = 0; j < n; j++) pq.offer(new Tuple(0, j, matrix[0][j]));
    for(int i = 0; i < k - 1; i++) { // 小根堆，去掉 k - 1 个堆顶元素，此时堆顶元素就是第 k 的数
        Tuple t = pq.poll();
        if(t.x == m - 1) continue;
        pq.offer(new Tuple(t.x + 1, t.y, matrix[t.x + 1][t.y]));
    }
    return pq.poll().val;
}

class Tuple implements Comparable<Tuple> {
    int x, y, val;
    public Tuple(int x, int y, int val) {
        this.x = x; this.y = y; this.val = val;
    }

    @Override
    public int compareTo(Tuple that) {
        return this.val - that.val;
    }
}
```

## 链表

**判断两个链表的交点**

[Leetcode : 160. Intersection of Two Linked Lists (Easy)](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)

```html
A:          a1 → a2
                  ↘
                    c1 → c2 → c3
                  ↗
B:    b1 → b2 → b3
```

要求：时间复杂度为 O(n) 空间复杂度为 O(1)

设 A 的长度为 a + c，B 的长度为 b + c，其中 c 为尾部公共部分长度，可知 a + c + b = b + c + a。

当访问 A 链表的指针访问到链表尾部时，令它从链表 B 的头部开始访问链表 B；同样地，当访问 B 链表的指针访问到链表尾部时，令它从链表 A 的头部开始访问链表 A。这样就能控制访问 A 和 B 两个链表的指针能同时访问到交点。

```python
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if(headA == null || headB == null) return null;
    ListNode l1 = headA, l2 = headB;
    while(l1 != l2){
        l1 = (l1 == null) ? headB : l1.next;
        l2 = (l2 == null) ? headA : l2.next;
    }
    return l1;
}
```

如果只是判断是否存在交点，那么就是另一个问题，即 [编程之美：3.6]() 的问题。有两种解法：把第一个链表的结尾连接到第二个链表的开头，看第二个链表是否存在环；或者直接比较第一个链表最后一个节点和第二个链表最后一个节点是否相同。

**链表反转**

[Leetcode : 206. Reverse Linked List (Easy)](https://leetcode.com/problems/reverse-linked-list/description/)

头插法能够按逆序构建链表。

```python
public ListNode reverseList(ListNode head) {
    ListNode newHead = null; // 设为 null，作为新链表的结尾
    while(head != null){
        ListNode nextNode = head.next;
        head.next = newHead;
        newHead = head;
        head = nextNode;
    }
    return newHead;
}
```

**归并两个有序的链表**

[Leetcode : 21. Merge Two Sorted Lists (Easy)](https://leetcode.com/problems/merge-two-sorted-lists/description/)

链表和树一样，可以用递归方式来定义：链表是空节点，或者有一个值和一个指向下一个链表的指针。因此很多链表问题可以用递归来处理。

```python
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l1 == null) return l2;
    if(l2 == null) return l1;
    ListNode newHead = null;
    if(l1.val < l2.val){
        newHead = l1;
        newHead.next = mergeTwoLists(l1.next, l2);
    } else{
        newHead = l2;
        newHead.next = mergeTwoLists(l1, l2.next);
    }
    return newHead;
}
```

**从有序链表中删除重复节点**

[Leetcode : 83. Remove Duplicates from Sorted List (Easy)](https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/)

```python
public ListNode deleteDuplicates(ListNode head) {
    if(head == null || head.next == null) return head;
    head.next = deleteDuplicates(head.next);
    return head.next != null && head.val == head.next.val ? head.next : head;
}
```

**回文链表**

[Leetcode : 234. Palindrome Linked List (Easy)](https://leetcode.com/problems/palindrome-linked-list/description/)

切成两半，把后半段反转，然后比较两半是否相等。

```python
public boolean isPalindrome(ListNode head) {
    if(head == null || head.next == null) return true;
    ListNode slow = head, fast = head.next;
    while(fast != null && fast.next != null){
        slow = slow.next;
        fast = fast.next.next;
    }

    if(fast != null){  // 偶数节点，让 slow 指向下一个节点
        slow = slow.next;
    }

    cut(head, slow); // 切成两个链表
    ListNode l1 = head, l2 = slow;
    l2 = reverse(l2);
    return isEqual(l1, l2);
}

private void cut(ListNode head, ListNode cutNode){
    while( head.next != cutNode ) head = head.next;
    head.next = null;
}

private ListNode reverse(ListNode head){
    ListNode newHead = null;
    while(head != null){
        ListNode nextNode = head.next;
        head.next = newHead;
        newHead = head;
        head = nextNode;
    }
    return newHead;
}

private boolean isEqual(ListNode l1, ListNode l2){
    while(l1 != null && l2 != null){
        if(l1.val != l2.val) return false;
        l1 = l1.next;
        l2 = l2.next;
    }
    return true;
}
```

**链表元素按奇偶聚集**

[Leetcode : 328. Odd Even Linked List (Medium)](https://leetcode.com/problems/odd-even-linked-list/description/)

```html
Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.
```

```python
public ListNode oddEvenList(ListNode head) {
    if (head == null) {
        return head;
    }
    ListNode odd = head, even = head.next, evenHead = even;
    while (even != null && even.next != null) {
        odd.next = odd.next.next;
        odd = odd.next;
        even.next = even.next.next;
        even = even.next;
    }
    odd.next = evenHead;
    return head;
}
```

## 树

### 递归

一棵树要么是空树，要么有两个指针，每个指针指向一棵树。树是一种递归结构，很多树的问题可以使用递归来处理。

**树的高度**

[Leetcode : 104. Maximum Depth of Binary Tree (Easy)](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)

```python
public int maxDepth(TreeNode root) {
    if(root == null) return 0;
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
}
```

**翻转树**

[Leetcode : 226. Invert Binary Tree (Easy)](https://leetcode.com/problems/invert-binary-tree/description/)

```python
public TreeNode invertTree(TreeNode root) {
    if(root == null) return null;
    TreeNode left = root.left; // 后面的操作会改变 left 指针，因此先保存下来
    root.left = invertTree(root.right);
    root.right = invertTree(left);
    return root;
}
```

**归并两棵树**

[Leetcode : 617. Merge Two Binary Trees (Easy)](https://leetcode.com/problems/merge-two-binary-trees/description/)

```html
Input:
       Tree 1                     Tree 2
          1                         2
         / \                       / \
        3   2                     1   3
       /                           \   \
      5                             4   7
Output:
Merged tree:
         3
        / \
       4   5
      / \   \
     5   4   7
```

```python
public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
    if(t1 == null && t2 == null) return null;
    if(t1 == null) return t2;
    if(t2 == null) return t1;
    TreeNode root = new TreeNode(t1.val + t2.val);
    root.left = mergeTrees(t1.left, t2.left);
    root.right = mergeTrees(t1.right, t2.right);
    return root;
}
```

**判断路径和是否等于一个数**

[Leetcdoe : 112. Path Sum (Easy)](https://leetcode.com/problems/path-sum/description/)

```html
Given the below binary tree and sum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
```

路径和定义为从 root 到 leaf 的所有节点的和

```python
public boolean hasPathSum(TreeNode root, int sum) {
    if(root == null) return false;
    if(root.left == null && root.right == null && root.val == sum) return true;
    return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
}
```

**统计路径和等于一个数的路径数量**

[Leetcode : 437. Path Sum III (Easy)](https://leetcode.com/problems/path-sum-iii/description/)

```html
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

路径不一定以 root 开头并以 leaf 结尾，但是必须连续

```python
public int pathSum(TreeNode root, int sum) {
    if(root == null) return 0;
    int ret = pathSumStartWithRoot(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    return ret;
}

private int pathSumStartWithRoot(TreeNode root, int sum){
    if(root == null) return 0;
    int ret = 0;
    if(root.val == sum) ret++;
    ret += pathSumStartWithRoot(root.left, sum - root.val) + pathSumStartWithRoot(root.right, sum - root.val);
    return ret;
}
```

**树的对称**

[Leetcode : 101. Symmetric Tree (Easy)](https://leetcode.com/problems/symmetric-tree/description/)

```html
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

```python
public boolean isSymmetric(TreeNode root) {
    if(root == null) return true;
    return isSymmetric(root.left, root.right);
}

private boolean isSymmetric(TreeNode t1, TreeNode t2){
    if(t1 == null && t2 == null) return true;
    if(t1 == null || t2 == null) return false;
    if(t1.val != t2.val) return false;
    return isSymmetric(t1.left, t2.right) && isSymmetric(t1.right, t2.left);
}
```

**平衡树**

[Leetcode : 110. Balanced Binary Tree (Easy)](https://leetcode.com/problems/balanced-binary-tree/description/)

```html
    3
   / \
  9  20
    /  \
   15   7
```

平衡树左右子树高度差都小于等于 1

```python
private boolean result = true;

public boolean isBalanced(TreeNode root) {
    maxDepth(root);
    return result;
}

public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    int l = maxDepth(root.left);
    int r = maxDepth(root.right);
    if (Math.abs(l - r) > 1) result = false;
    return 1 + Math.max(l, r);
}
```

**最小路径**

[Leetcode : 111. Minimum Depth of Binary Tree (Easy)](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/)

树的根节点到叶子节点的最小路径长度

```python
public int minDepth(TreeNode root) {
    if(root == null) return 0;
    int left = minDepth(root.left);
    int right = minDepth(root.right);
    if(left == 0 || right == 0) return left + right + 1;
    return Math.min(left, right) + 1;
}
```

**统计左叶子节点的和**

[Leetcode : 404. Sum of Left Leaves (Easy)](https://leetcode.com/problems/sum-of-left-leaves/description/)

```html
    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
```

```python
public int sumOfLeftLeaves(TreeNode root) {
    if(root == null) {
        return 0;
    }
    if(isLeaf(root.left)) {
        return root.left.val +  sumOfLeftLeaves(root.right);
    }
    return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
}

private boolean isLeaf(TreeNode node){
    if(node == null) {
        return false;
    }
    return node.left == null && node.right == null;
}
```

**修剪二叉查找树**

[Leetcode : 669. Trim a Binary Search Tree (Easy)](https://leetcode.com/problems/trim-a-binary-search-tree/description/)

```html
Input:
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output:
      3
     /
   2
  /
 1
```

二叉查找树（BST）：根节点大于等于左子树所有节点，小于等于右子树所有节点。

只保留值在 L \~ R 之间的节点

```python
public TreeNode trimBST(TreeNode root, int L, int R) {
    if(root == null) return null;
    if(root.val > R) return trimBST(root.left, L, R);
    if(root.val < L) return trimBST(root.right, L, R);
    root.left = trimBST(root.left, L, R);
    root.right = trimBST(root.right, L, R);
    return root;
}
```

**子树**

[Leetcode : 572. Subtree of Another Tree (Easy)](https://leetcode.com/problems/subtree-of-another-tree/description/)

```html
Given tree s:
     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
```

```python
public boolean isSubtree(TreeNode s, TreeNode t) {
    if(s == null && t == null) return true;
    if(s == null || t == null) return false;
    if(s.val == t.val && isSame(s, t)) return true;
    return isSubtree(s.left, t) || isSubtree(s.right, t);
}

private boolean isSame(TreeNode s, TreeNode t){
    if(s == null && t == null) return true;
    if(s == null || t == null) return false;
    if(s.val != t.val) return false;
    return isSame(s.left, t.left) && isSame(s.right, t.right);
}
```

**从有序数组中构造二叉查找树**

[Leetcode : 108. Convert Sorted Array to Binary Search Tree (Easy)](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/)

```python
public TreeNode sortedArrayToBST(int[] nums) {
    return toBST(nums, 0, nums.length - 1);
}

private TreeNode toBST(int[] nums, int sIdx, int eIdx){
    if(sIdx > eIdx) return null;
    int mIdx = (sIdx + eIdx) / 2;
    TreeNode root = new TreeNode(nums[mIdx]);
    root.left =  toBST(nums, sIdx, mIdx - 1);
    root.right = toBST(nums, mIdx + 1, eIdx);
    return root;
}
```

**两节点的最长路径**

```html
Input:

         1
        / \
       2  3
      / \
      4  5

Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
```

[Leetcode : 543. Diameter of Binary Tree (Easy)](https://leetcode.com/problems/diameter-of-binary-tree/description/)

```python
private int max = 0;

public int diameterOfBinaryTree(TreeNode root) {
    depth(root);
    return max;
}

private int depth(TreeNode root) {
    if (root == null) return 0;
    int leftDepth = depth(root.left);
    int rightDepth = depth(root.right);
    max = Math.max(max, leftDepth + rightDepth);
    return Math.max(leftDepth, rightDepth) + 1;
}
```

**找出二叉树中第二小的节点**

[Leetcode : 671. Second Minimum Node In a Binary Tree (Easy)](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/)

```html
Input:
   2
  / \
 2   5
    / \
    5  7

Output: 5
```

一个节点要么具有 0 个或 2 个子节点，如果有子节点，那么根节点是最小的节点。

```python
public int findSecondMinimumValue(TreeNode root) {
    if(root == null) return -1;
    if(root.left == null && root.right == null) return -1;
    int leftVal = root.left.val;
    int rightVal = root.right.val;
    if(leftVal == root.val) leftVal = findSecondMinimumValue(root.left);
    if(rightVal == root.val) rightVal = findSecondMinimumValue(root.right);
    if(leftVal != -1 && rightVal != -1) return Math.min(leftVal, rightVal);
    if(leftVal != -1) return leftVal;
    return rightVal;
}
```

**二叉查找树的最近公共祖先**

[Leetcode : 235. Lowest Common Ancestor of a Binary Search Tree (Easy)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)

```html
        _______6______
       /              \
    ___2__          ___8__
   /      \        /      \
   0      _4       7       9
         /  \
         3   5
For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
```

```python
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if(root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
    if(root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
    return root;
}
```

**二叉树的最近公共祖先**

[Leetcode : 236. Lowest Common Ancestor of a Binary Tree (Medium) ](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/)

```html
       _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
```

```python
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    return left == null ? right : right == null ? left : root;
}
```

**相同节点值的最大路径长度**

[Leetcode : 687. Longest Univalue Path (Easy)](https://leetcode.com/problems/longest-univalue-path/)

```html
             1
            / \
           4   5
          / \   \
         4  4    5

Output : 2
```

```python
private int path = 0;
public int longestUnivaluePath(TreeNode root) {
    dfs(root);
    return path;
}

private int dfs(TreeNode root){
    if(root == null) return 0;
    int left = dfs(root.left);
    int right = dfs(root.right);
    int leftPath = root.left != null && root.left.val == root.val ? left + 1 : 0;
    int rightPath = root.right != null && root.right.val == root.val ? right + 1 : 0;
    path = Math.max(path, leftPath + rightPath);
    return Math.max(leftPath, rightPath);
}
```

**间隔遍历**

[Leetcode : 337. House Robber III (Medium)](https://leetcode.com/problems/house-robber-iii/description/)

```html
     3
    / \
   2   3
    \   \
     3   1
Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```

```python
public int rob(TreeNode root) {
    if (root == null) return 0;
    int val1 = root.val;
    if (root.left != null) {
        val1 += rob(root.left.left) + rob(root.left.right);
    }
    if (root.right != null) {
        val1 += rob(root.right.left) + rob(root.right.right);
    }
    int val2 = rob(root.left) + rob(root.right);
    return Math.max(val1, val2);
}
```

### 层次遍历

使用 BFS 进行层次遍历。不需要使用两个队列来分别存储当前层的节点和下一层的节点，因为在开始遍历一层的节点时，当前队列中的节点数就是当前层的节点数，只要控制遍历这么多节点数，就能保证这次遍历的都是当前层的节点。

**一棵树每层节点的平均数**

[637. Average of Levels in Binary Tree (Easy)](https://leetcode.com/problems/average-of-levels-in-binary-tree/description/)

```python
public List<Double> averageOfLevels(TreeNode root) {
    List<Double> ret = new ArrayList<>();
    if(root == null) return ret;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while(!queue.isEmpty()){
        int cnt = queue.size();
        double sum = 0;
        for(int i = 0; i < cnt; i++){
            TreeNode node = queue.poll();
            sum += node.val;
            if(node.left != null) queue.add(node.left);
            if(node.right != null) queue.add(node.right);
        }
        ret.add(sum / cnt);
    }
    return ret;
}
```

**得到左下角的节点**

[Leetcode : 513. Find Bottom Left Tree Value (Easy)](https://leetcode.com/problems/find-bottom-left-tree-value/description/)

```html
Input:

        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

Output:
7
```

```python
public int findBottomLeftValue(TreeNode root) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while(!queue.isEmpty()){
        root = queue.poll();
        if(root.right != null) queue.add(root.right);
        if(root.left != null) queue.add(root.left);
    }
    return root.val;
}
```

### 前中后序遍历

```html
   1
  / \
  2  3
 / \  \
4  5  6
```

层次遍历顺序：[1 2 3 4 5 6]
前序遍历顺序：[1 2 4 5 3 6]
中序遍历顺序：[4 2 5 1 3 6]
后序遍历顺序：[4 5 2 6 3 1]

层次遍历使用 BFS 实现，利用的就是 BFS 一层一层遍历的特性；而前序、中序、后序遍历利用了 DFS 实现。

前序、中序、后序遍只是在对节点访问的顺序有一点不同，其它都相同。

① 前序

```python
void dfs(TreeNode root){
    visit(root);
    dfs(root.left);
    dfs(root.right);
}
```

② 中序

```python
void dfs(TreeNode root){
    dfs(root.left);
    visit(root);
    dfs(root.right);
}
```

③ 后序

```python
void dfs(TreeNode root){
    dfs(root.left);
    dfs(root.right);
    visit(root);
}
```

**非递归实现二叉树的前序遍历**

[Leetcode : 144. Binary Tree Preorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

```python
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    if (root == null) return ret;
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        ret.add(node.val);
        if (node.right != null) stack.push(node.right);
        if (node.left != null) stack.push(node.left); // 先添加右子树再添加左子树，这样是为了让左子树在栈顶
    }
    return ret;
}
```

**非递归实现二叉树的后序遍历**

[Leetcode : 145. Binary Tree Postorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)

前序遍历为 root -> left -> right，后序遍历为 left -> right -> root，可以修改前序遍历成为 root -> right -> left，那么这个顺序就和后序遍历正好相反。

```python
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    if (root == null) return ret;
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        ret.add(node.val);
        if (node.left != null) stack.push(node.left);
        if (node.right != null) stack.push(node.right);
    }
    Collections.reverse(ret);
    return ret;
}
```

**非递归实现二叉树的中序遍历**

[Leetcode : 94. Binary Tree Inorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)

```python
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = root;
    while(cur != null || !stack.isEmpty()) {
        while(cur != null) { // 模拟递归栈的不断深入
            stack.add(cur);
            cur = cur.left;
        }
        TreeNode node = stack.pop();
        ret.add(node.val);
        cur = node.right;
    }
    return ret;
}
```

### BST

主要利用 BST 中序遍历有序的特点。

**在 BST 中寻找两个节点，使它们的和为一个给定值**

[653. Two Sum IV - Input is a BST (Easy)](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/)

```html
Input:
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True
```

使用中序遍历得到有序数组之后，再利用双指针对数组进行查找。

应该注意到，这一题不能用分别在左右子树两部分来处理这种思想，因为两个待求的节点可能分别在左右子树中。

```python
public boolean findTarget(TreeNode root, int k) {
    List<Integer> nums = new ArrayList<>();
    inOrder(root, nums);
    int i = 0, j = nums.size() - 1;
    while(i < j){
        int sum = nums.get(i) + nums.get(j);
        if(sum == k) return true;
        if(sum < k) i++;
        else j--;
    }
    return false;
}

private void inOrder(TreeNode root, List<Integer> nums){
    if(root == null) return;
    inOrder(root.left, nums);
    nums.add(root.val);
    inOrder(root.right, nums);
}
```

**在 BST 中查找两个节点之差的最小绝对值**

[Leetcode : 530. Minimum Absolute Difference in BST (Easy)](https://leetcode.com/problems/minimum-absolute-difference-in-bst/description/)

```html
Input:
   1
    \
     3
    /
   2

Output:
1
```

利用 BST 的中序遍历为有序的性质，计算中序遍历中临近的两个节点之差的绝对值，取最小值。

```python
private int minDiff = Integer.MAX_VALUE;
private int preVal = -1;

public int getMinimumDifference(TreeNode root) {
    inorder(root);
    return minDiff;
}

private void inorder(TreeNode node){
    if(node == null) return;
    inorder(node.left);
    if(preVal != -1) minDiff = Math.min(minDiff, Math.abs(node.val - preVal));
    preVal = node.val;
    inorder(node.right);
}
```

**把 BST 每个节点的值都加上比它大的节点的值**

[Leetcode : Convert BST to Greater Tree (Easy)](https://leetcode.com/problems/convert-bst-to-greater-tree/description/)

```html
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

先遍历右子树。

```python
private int sum = 0;

public TreeNode convertBST(TreeNode root) {
    traver(root);
    return root;
}

private void traver(TreeNode root) {
    if (root == null) {
        return;
    }
    if (root.right != null) {
        traver(root.right);
    }
    sum += root.val;
    root.val = sum;
    if (root.left != null) {
        traver(root.left);
    }
}
```

**寻找 BST 中出现次数最多的节点**

[Leetcode : 501. Find Mode in Binary Search Tree (Easy)](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/)

```html
   1
    \
     2
    /
   2
return [2].
```

```python
private int cnt = 1;
private int maxCnt = 1;
private TreeNode preNode = null;
private List<Integer> list;

public int[] findMode(TreeNode root) {
    list = new ArrayList<>();
    inorder(root);
    int[] ret = new int[list.size()];
    int idx = 0;
    for(int num : list){
        ret[idx++] = num;
    }
    return ret;
}

private void inorder(TreeNode node){
    if(node == null) return;
    inorder(node.left);
    if(preNode != null){
        if(preNode.val == node.val) cnt++;
        else cnt = 1;
    }
    if(cnt > maxCnt){
        maxCnt = cnt;
        list.clear();
        list.add(node.val);
    } else if(cnt == maxCnt){
        list.add(node.val);
    }
    preNode = node;
    inorder(node.right);
}
```

**寻找 BST 的第 k 个元素**

[Leetcode : 230. Kth Smallest Element in a BST (Medium)](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)

递归解法：

```python
public int kthSmallest(TreeNode root, int k) {
    int leftCnt = count(root.left);
    if(leftCnt == k - 1) return root.val;
    if(leftCnt > k - 1) return kthSmallest(root.left, k);
    return kthSmallest(root.right, k - leftCnt - 1);
}

private int count(TreeNode node) {
    if(node == null) return 0;
    return 1 + count(node.left) + count(node.right);
}
```

中序遍历解法：

```python
private int cnt = 0;
private int val;

public int kthSmallest(TreeNode root, int k) {
    inorder(root, k);
    return val;
}

private void inorder(TreeNode node, int k) {
    if(node == null) return;
    inorder(node.left, k);
    cnt++;
    if(cnt == k) {
        val = node.val;
        return;
    }
    inorder(node.right, k);
}
```

### Trie

<div align="center"> <img src="../pics//5c638d59-d4ae-4ba4-ad44-80bdc30f38dd.jpg"/> </div><br>

Trie，又称前缀树或字典树，用于判断字符串是否存在或者是否具有某种字符串前缀。

**实现一个 Trie**

[Leetcode : 208. Implement Trie (Prefix Tree) (Medium)](https://leetcode.com/problems/implement-trie-prefix-tree/description/)

```python
class Trie {

    private class Node{
        Node[] childs = new Node[26];
        boolean isLeaf;
    }

    private Node root = new Node();

    /** Initialize your data structure here. */
    public Trie() {
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        int idx = word.charAt(0) - 'a';
        insert(word, root);
    }

    private void insert(String word, Node node){
        int idx = word.charAt(0) - 'a';
        if(node.childs[idx] == null){
            node.childs[idx] = new Node();
        }
        if(word.length() == 1) node.childs[idx].isLeaf = true;
        else insert(word.substring(1), node.childs[idx]);
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        return search(word, root);
    }

    private boolean search(String word, Node node){
        if(node == null) return false;
        int idx = word.charAt(0) - 'a';
        if(node.childs[idx] == null) return false;
        if(word.length() == 1) return node.childs[idx].isLeaf;
        return search(word.substring(1), node.childs[idx]);
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        return startWith(prefix, root);
    }

    private boolean startWith(String prefix, Node node){
        if(node == null) return false;
        if(prefix.length() == 0) return true;
        int idx = prefix.charAt(0) - 'a';
        return startWith(prefix.substring(1), node.childs[idx]);
    }
}
```

**实现一个 Trie，用来求前缀和**

[Leetcode : 677. Map Sum Pairs (Medium)](https://leetcode.com/problems/map-sum-pairs/description/)

```html
Input: insert("apple", 3), Output: Null
Input: sum("ap"), Output: 3
Input: insert("app", 2), Output: Null
Input: sum("ap"), Output: 5
```

```python
class MapSum {
    private class Trie {
        int val;
        Map<Character, Trie> childs;
        boolean isWord;

        Trie() {
            childs = new HashMap<>();
        }
    }

    private Trie root;

    public MapSum() {
        root = new Trie();
    }

    public void insert(String key, int val) {
        Trie cur = root;
        for(char c : key.toCharArray()) {
            if(!cur.childs.containsKey(c)) {
                Trie next = new Trie();
                cur.childs.put(c, next);
            }
            cur = cur.childs.get(c);
        }
        cur.val = val;
        cur.isWord = true;
    }

    public int sum(String prefix) {
        Trie cur = root;
        for(char c : prefix.toCharArray()) {
            if(!cur.childs.containsKey(c)) return 0;
            cur = cur.childs.get(c);
        }
        return dfs(cur);
    }

    private int dfs(Trie cur) {
        int sum = 0;
        if(cur.isWord) {
            sum += cur.val;
        }
        for(Trie next : cur.childs.values()) {
            sum += dfs(next);
        }
        return sum;
    }
}
```

## 图

## 位运算

**1. 基本原理**

0s 表示 一串 0 ，1s 表示一串 1。

```
x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x = 0       x & x = x       x | x = x
```

① 利用 x ^ 1s = \~x 的特点，可以将位级表示翻转；利用 x ^ x = 0 的特点，可以将三个数中重复的两个数去除，只留下另一个数；
② 利用 x & 0s = 0 和 x & 1s = x 的特点，可以实现掩码操作。一个数 num 与 mask ：00111100 进行位与操作，只保留 num 中与 mask 的 1 部分相对应的位；
③ 利用 x | 0s = x 和 x | 1s = 1s 的特点，可以实现设置操作。一个数 num 与 mask：00111100 进行位或操作，将 num 中与 mask 的 1 部分相对应的位都设置为 1 。

\>\> n 为算术右移，相当于除以 2<sup>n</sup>；
\>\>\> n 为无符号右移，左边会补上 0。
&lt;&lt; n 为算术左移，相当于乘以 2<sup>n</sup>。

n&(n-1) 该位运算是去除 n 的位级表示中最低的那一位。例如对于二进制表示 10110 **100** ，减去 1 得到 10110**011**，这两个数相与得到 10110**000**。

n-n&(\~n+1) 概运算是去除 n 的位级表示中最高的那一位。

n&(-n) 该运算得到 n 的位级表示中最低的那一位。-n 得到 n 的反码加 1，对于二进制表示 10110 **100** ，-n 得到 01001**100**，相与得到 00000**100**

**2. mask 计算**

要获取 111111111，将 0 取反即可，\~0。

要得到只有第 i 位为 1 的 mask，将 1 向左移动 i 位即可，1&lt;&lt;i 。例如 1&lt;&lt;5 得到只有第 5 位为 1 的 mask ：00010000。

要得到 1 到 i 位为 1 的 mask，1&lt;&lt;(i+1)-1 即可，例如将 1&lt;&lt;(4+1)-1 = 00010000-1 = 00001111。

要得到 1 到 i 位为 0 的 mask，只需将 1 到 i 位为 1 的 mask 取反，即 \~(1&lt;&lt;(i+1)-1)。

**3. 位操作举例**

① 获取第 i 位

num & 00010000 != 0

```python
(num & (1 << i)) != 0;
```

② 将第 i 位设置为 1

num | 00010000

```python
num | (1 << i);
```

③ 将第 i 位清除为 0

num & 11101111

```python
num & (~(1 << i))
```

④ 将最高位到第 i 位清除为 0

num & 00001111

```python
num & ((1 << i) - 1);
```

⑤ 将第 0 位到第 i 位清除为 0

num & 11110000

```python
num & (~((1 << (i+1)) - 1));
```

⑥ 将第 i 位设置为 0 或者 1

先将第 i 位清零，然后将 v 左移 i 位，执行“位或”运算。

```python
(num & (1 << i)) | (v << i);
```

**4. Java 中的位操作**

```html
static int Integer.bitCount()            // 统计 1 的数量
static int Integer.highestOneBit()       // 获得最高位
static String toBinaryString(int i)      // 转换位二进制表示的字符串
```

**统计两个数的二进制表示有多少位不同**

[Leetcode : 461. Hamming Distance (Easy)](https://leetcode.com/problems/hamming-distance/)

对两个数进行异或操作，不同的那一位结果为 1 ，统计有多少个 1 即可。

```python
public int hammingDistance(int x, int y) {
    int z = x ^ y;
    int cnt = 0;
    while(z != 0){
        if((z & 1) == 1) cnt++;
        z = z >> 1;
    }
    return cnt;
}
```

可以使用 Integer.bitcount() 来统计 1 个的个数。

```python
public int hammingDistance(int x, int y) {
    return Integer.bitCount(x ^ y);
}
```

**翻转一个数的比特位**

[Leetcode : 190. Reverse Bits (Easy)](https://leetcode.com/problems/reverse-bits/description/)

```python
public int reverseBits(int n) {
    int ret = 0;
    for(int i = 0; i < 32; i++){
        ret <<= 1;
        ret |= (n & 1);
        n >>>= 1;
    }
    return ret;
}
```

**不用额外变量交换两个整数**

[程序员代码面试指南 ：P317](#)

```python
a = a ^ b;
b = a ^ b;
a = a ^ b;
```

将 c = a ^ b，那么 b ^ c = b ^ b ^ a = a，a ^ c = a ^ a ^ b = b。

**判断一个数是不是 4 的 n 次方**

[Leetcode : 342. Power of Four (Easy)](https://leetcode.com/problems/power-of-four/)

该数二进制表示有且只有一个奇数位为 1 ，其余的都为 0 ，例如 16 ： 10000。可以每次把 1 向左移动 2 位，就能构造出这种数字，然后比较构造出来的数与要判断的数是否相同。

```python
public boolean isPowerOfFour(int num) {
    int i = 1;
    while(i > 0){
        if(i == num) return true;
        i = i << 2;
    }
    return false;
}
```

也可以用 Java 的 Integer.toString() 方法将该数转换为 4 进制形式的字符串，然后判断字符串是否以 1 开头。

```python
public boolean isPowerOfFour(int num) {
    return Integer.toString(num, 4).matches("10*");
}
```

**判断一个数是不是 2 的 n 次方**

[Leetcode : 231. Power of Two (Easy)](https://leetcode.com/problems/power-of-two/description/)

同样可以用 Power of Four 的方法，但是 2 的 n 次方更特殊，它的二进制表示只有一个 1 存在。

```python
public boolean isPowerOfTwo(int n) {
    return n > 0 && Integer.bitCount(n) == 1;
}
```

利用 1000 & 0111 == 0 这种性质，得到以下解法：

```python
public boolean isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

**数组中唯一一个不重复的元素**

[Leetcode : 136. Single Number (Easy)](https://leetcode.com/problems/single-number/description/)

两个相同的数异或的结果为 0，对所有数进行异或操作，最后的结果就是单独出现的那个数。

类似的有：[Leetcode : 389. Find the Difference (Easy)](https://leetcode.com/problems/find-the-difference/description/)，两个字符串仅有一个字符不相同，使用异或操作可以以 O(1) 的空间复杂度来求解，而不需要使用 HashSet。

```python
public int singleNumber(int[] nums) {
    int ret = 0;
    for(int n : nums) ret = ret ^ n;
    return ret;
}
```

**数组中不重复的两个元素**

[Leetcode : 260. Single Number III (Medium)](https://leetcode.com/problems/single-number-iii/description/)

两个不相等的元素在位级表示上必定会有一位存在不同。

将数组的所有元素异或得到的结果为不存在重复的两个元素异或的结果。

diff &= -diff 得到出 diff 最右侧不为 0 的位，也就是不存在重复的两个元素在位级表示上最右侧不同的那一位，利用这一位就可以将两个元素区分开来。

```python
public int[] singleNumber(int[] nums) {
    int diff = 0;
    for(int num : nums) diff ^= num;
    // 得到最右一位
    diff &= -diff;
    int[] ret = new int[2];
    for(int num : nums) {
        if((num & diff) == 0) ret[0] ^= num;
        else ret[1] ^= num;
    }
    return ret;
}
```

**判断一个数的位级表示是否不会出现连续的 0 和 1**

[Leetcode : 693. Binary Number with Alternating Bits (Easy)](https://leetcode.com/problems/binary-number-with-alternating-bits/description/)

对于 10101 这种位级表示的数，把它向右移动 1 位得到 1010 ，这两个数每个位都不同，因此异或得到的结果为 11111。

```python
public boolean hasAlternatingBits(int n) {
    int a = (n ^ (n >> 1));
    return (a & (a + 1)) == 0;
}
```

**求一个数的补码**

[Leetcode : 476. Number Complement (Easy)](https://leetcode.com/problems/number-complement/description/)

不考虑二进制表示中的首 0 部分

对于 00000101，要求补码可以将它与 00000111 进行异或操作。那么问题就转换为求掩码 00000111。

```python
public int findComplement(int num) {
    if(num == 0) return 1;
    int mask = 1 << 30;
    while((num & mask) == 0) mask >>= 1;
    mask = (mask << 1) - 1;
    return num ^ mask;
}
```

可以利用 Java 的 Integer.highestOneBit() 方法来获得含有首 1 的数。

```python
public int findComplement(int num) {
    if(num == 0) return 1;
    int mask = Integer.highestOneBit(num);
    mask = (mask << 1) - 1;
    return num ^ mask;
}
```

对于 10000000 这样的数要扩展成 11111111，可以利用以下方法：

```html
mask |= mask >> 1    11000000
mask |= mask >> 2    11110000
mask |= mask >> 4    11111111
```

```python
public int findComplement(int num) {
    int mask = num;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    return (mask ^ num);
}
```

**实现整数的加法**

[Leetcode : 371. Sum of Two Integers (Easy)](https://leetcode.com/problems/sum-of-two-integers/description/)

a ^ b 表示没有考虑进位的情况下两数的和，(a & b) << 1 就是进位。递归会终止的原因是 (a & b) << 1 最右边会多一个 0，那么继续递归，进位最右边的 0 会慢慢增多，最后进位会变为 0，递归终止。

```python
public int getSum(int a, int b) {
    return b == 0 ? a : getSum((a ^ b), (a & b) << 1);
}
```

**字符串数组最大乘积**

[Leetcode : 318. Maximum Product of Word Lengths (Medium)](https://leetcode.com/problems/maximum-product-of-word-lengths/description/)

题目描述：字符串数组的字符串只含有小写字符。求解字符串数组中两个字符串长度的最大乘积，要求这两个字符串不能含有相同字符。

解题思路：本题主要问题是判断两个字符串是否含相同字符，由于字符串只含有小写字符，总共 26 位，因此可以用一个 32 位的整数来存储每个字符是否出现过。

```python
public int maxProduct(String[] words) {
    int n = words.length;
    if (n == 0) return 0;
    int[] val = new int[n];
    for (int i = 0; i < n; i++) {
        for (char c : words[i].toCharArray()) {
            val[i] |= 1 << (c - 'a');
        }
    }
    int ret = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((val[i] & val[j]) == 0) {
                ret = Math.max(ret, words[i].length() * words[j].length());
            }
        }
    }
    return ret;
}
```
