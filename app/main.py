from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import re

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Problem data model
class Problem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    difficulty: str
    solution_code: str
    example_input: str
    example_output: str

# Database of problems
PROBLEMS_DB = {
    "two-sum": Problem(
        id="two-sum",
        title="Two Sum",
        description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        category="Array",
        difficulty="Easy",
        solution_code="""class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in hashmap:
                return [hashmap[complement], i]
            hashmap[num] = i
        return []""",
        example_input="[2,7,11,15], 9",
        example_output="[0, 1]"
    ),
    "valid-palindrome": Problem(
        id="valid-palindrome",
        title="Valid Palindrome",
        description="Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.",
        category="Strings",
        difficulty="Easy",
        solution_code="""class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join(c for c in s if c.isalnum()).lower()
        return s == s[::-1]""",
        example_input='"A man, a plan, a canal: Panama"',
        example_output="True"
    ),
    "merge-two-sorted-lists": Problem(
        id="merge-two-sorted-lists",
        title="Merge Two Sorted Lists",
        description="Merge two sorted linked lists and return it as a sorted list.",
        category="Linked List",
        difficulty="Easy",
        solution_code="""class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        current = dummy
    
        while l1 and l2:
            if l1.val < l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
    
        current.next = l1 if l1 else l2
        return dummy.next""",
        example_input="[1,2,4], [1,3,4]",
        example_output="[1,1,2,3,4,4]"
    ),
    "validate-binary-search-tree": Problem(
        id="validate-binary-search-tree",
        title="Validate Binary Search Tree",
        description="Given the root of a binary tree, determine if it is a valid binary search tree (BST).",
        category="Trees",
        difficulty="Easy",
        solution_code="""class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True
            val = node.val
            if val <= lower or val >= upper:
                return False
            return helper(node.right, val, upper) and helper(node.left, lower, val)
        return helper(root)""",
        example_input="[2,1,3]",
        example_output="True"
    ),
    "first-bad-version": Problem(
        id="first-bad-version",
        title="First Bad Version",
        description="Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one.",
        category="Sorting and Searching",
        difficulty="Easy",
        solution_code="""class Solution:
    def firstBadVersion(self, n: int) -> int:
    left, right = 1, n
    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left""",
        example_input="5, 4",
        example_output="4"
    ),
    "maximum-subarray": Problem(
        id="maximum-subarray",
        title="Maximum Subarray",
        description="Given an integer array nums, find the contiguous subarray which has the largest sum and return its sum.",
        category="Dynamic Programming",
        difficulty="Easy",
        solution_code="""class Solution:
    def maxSubArray(nums: List[int]) -> int:
        max_current = max_global = nums[0]
        for num in nums[1:]:
            max_current = max(num, max_current + num)
            max_global = max(max_global, max_current)
        return max_global""",
        example_input="[-2,1,-3,4,-1,2,1,-5,4]",
        example_output="6"
    ),
    "min-stack": Problem(
        id="min-stack",
        title="Min Stack",
        description="Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.",
        category="Design",
        difficulty="Easy",
        solution_code="""class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]""",
        example_input='["MinStack","push","push","push","getMin","pop","top","getMin"]\n[[],[-2],[0],[-3],[],[],[],[]]',
        example_output="[null,null,null,null,-3,null,0,-2]"
    ),
    "roman-to-integer": Problem(
        id="roman-to-integer",
        title="Roman to Integer",
        description="Given a roman numeral, convert it to an integer.",
        category="Math",
        difficulty="Easy",
        solution_code="""class Solution:
    def romanToInt(s: str) -> int:
        roman = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        total = 0
        for i in range(len(s)):
            if i < len(s)-1 and roman[s[i]] < roman[s[i+1]]:
                total -= roman[s[i]]
            else:
                total += roman[s[i]]
        return total""",
        example_input='"LVIII"',
        example_output="58"
    ),
    "number-of-1-bits": Problem(
        id="number-of-1-bits",
        title="Number of 1 Bits",
        description="Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).",
        category="Others",
        difficulty="Easy",
        solution_code="""class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        for i in range(32):
            if (n >> i) & 1:
                ans += 1
        return ans""",
        example_input="00000000000000000000000000001011",
        example_output="3"
    )
}

# Solutions for problems
def solve_two_sum(nums: List[int], target: int) -> List[int]:
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []

def solve_valid_palindrome(s: str) -> bool:
    s = ''.join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]

def solve_merge_two_sorted_lists(l1: List[int], l2: List[int]) -> List[int]:
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    def create_list(arr):
        if not arr:
            return None
        head = ListNode(arr[0])
        current = head
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        return head
    
    def list_to_array(node):
        arr = []
        while node:
            arr.append(node.val)
            node = node.next
        return arr
    
    list1 = create_list(l1)
    list2 = create_list(l2)
    
    dummy = ListNode()
    current = dummy
    
    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    current.next = list1 if list1 else list2
    return list_to_array(dummy.next)

def solve_validate_binary_search_tree(tree: List[Optional[int]]) -> bool:
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    def build_tree(arr, i=0):
        if i >= len(arr) or arr[i] is None:
            return None
        root = TreeNode(arr[i])
        root.left = build_tree(arr, 2*i + 1)
        root.right = build_tree(arr, 2*i + 2)
        return root
    
    def helper(node, lower=float('-inf'), upper=float('inf')):
        if not node:
            return True
        val = node.val
        if val <= lower or val >= upper:
            return False
        return helper(node.right, val, upper) and helper(node.left, lower, val)
    
    root = build_tree(tree)
    return helper(root)

def solve_first_bad_version(n: int, bad: int) -> int:
    def isBadVersion(version):
        return version >= bad
    left, right = 1, n
    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left

def solve_max_subarray(nums: List[int]) -> int:
    max_current = max_global = nums[0]
    for num in nums[1:]:
        max_current = max(num, max_current + num)
        max_global = max(max_global, max_current)
    return max_global

def solve_min_stack(operations: List[str], values: List[List[int]]) -> List[Optional[int]]:
    result = []
    min_stack = None
    
    for op, val in zip(operations, values):
        if op == "MinStack":
            class MinStack:
                def __init__(self):
                    self.stack = []
                    self.min_stack = []

                def push(self, val: int) -> None:
                    self.stack.append(val)
                    if not self.min_stack or val <= self.min_stack[-1]:
                        self.min_stack.append(val)

                def pop(self) -> None:
                    if self.stack.pop() == self.min_stack[-1]:
                        self.min_stack.pop()

                def top(self) -> int:
                    return self.stack[-1]

                def getMin(self) -> int:
                    return self.min_stack[-1]
            
            min_stack = MinStack()
            result.append(None)
        elif op == "push":
            min_stack.push(val[0])
            result.append(None)
        elif op == "pop":
            min_stack.pop()
            result.append(None)
        elif op == "top":
            result.append(min_stack.top())
        elif op == "getMin":
            result.append(min_stack.getMin())
    
    return result

def solve_roman_to_integer(s: str) -> int:
    roman = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    total = 0
    for i in range(len(s)):
        if i < len(s)-1 and roman[s[i]] < roman[s[i+1]]:
            total -= roman[s[i]]
        else:
            total += roman[s[i]]
    return total

def solve_number_of_1_bits(n: int) -> int:
    try:
        ans = 0
        for i in range(32):
            if (n >> i) & 1:
                ans += 1
        return ans
    
    except ValueError as e:
        raise ValueError("Invalid input. Please enter a 32-bit unsigned integer in decimal or binary format (e.g., 11 or 0b1011)") from e

# Map problem IDs to their solution functions
SOLVERS = {
    "two-sum": solve_two_sum,
    "valid-palindrome": solve_valid_palindrome,
    "merge-two-sorted-lists": solve_merge_two_sorted_lists,
    "validate-binary-search-tree": solve_validate_binary_search_tree,
    "first-bad-version": solve_first_bad_version,
    "maximum-subarray": solve_max_subarray,
    "min-stack": solve_min_stack,
    "roman-to-integer": solve_roman_to_integer,
    "number-of-1-bits": solve_number_of_1_bits
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Group problems by category
    categories = {}
    for problem in PROBLEMS_DB.values():
        if problem.category not in categories:
            categories[problem.category] = []
        categories[problem.category].append(problem)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categories": categories
    })

@app.get("/problem/{problem_id}", response_class=HTMLResponse)
async def read_problem(request: Request, problem_id: str):
    problem = PROBLEMS_DB.get(problem_id)
    if not problem:
        return RedirectResponse("/")
    
    return templates.TemplateResponse("problem.html", {
        "request": request,
        "problem": problem
    })

@app.post("/problem/{problem_id}", response_class=HTMLResponse)
async def solve_problem(request: Request, problem_id: str, user_input: str = Form(...)):
    problem = PROBLEMS_DB.get(problem_id)
    if not problem:
        return RedirectResponse("/")
    
    solver = SOLVERS.get(problem_id)
    if not solver:
        return templates.TemplateResponse("problem.html", {
            "request": request,
            "problem": problem,
            "error": "Solution function not implemented yet."
        })
    
    try:
        # Special handling for MinStack problem
        if problem_id == "min-stack":
            parts = [part.strip() for part in user_input.split("\n")]
            if len(parts) != 2:
                raise ValueError("Input should have two lines: operations and values")
            
            operations = eval(parts[0])
            values = eval(parts[1])
            result = solver(operations, values)
        else:
            # Parse input (this is a simplified approach - in real app you'd need better parsing)
            parsed_input = eval(user_input)
            
            # Handle single argument vs multiple arguments
            if isinstance(parsed_input, tuple):
                result = solver(*parsed_input)
            else:
                result = solver(parsed_input)
            
        return templates.TemplateResponse("problem.html", {
            "request": request,
            "problem": problem,
            "user_input": user_input,
            "result": result
        })
    except Exception as e:
        return templates.TemplateResponse("problem.html", {
            "request": request,
            "problem": problem,
            "user_input": user_input,
            "error": f"Error processing input: {str(e)}"
        })