# Javascript Coding Style Document

## 1. Background
This document serves to define our coding standards code in the JavaScript programming language. This document focuses primarily on the hard-and-fast rules that we follow universally, and avoids giving advice that isn't clearly enforceable (whether by human or tool).

## 2. Rules
### 2.1 File Names
File names must be all lowercase and may include underscores or dashes (hyphen) but no additional punctuation. Follow the convention that your project uses. 

### 2.2 File Encoding
Source files are encoded in UTF-8. 

### 2.3 Special Characters
#### 2.3.1 Whitespace Characters
Asside form the line terminator sequence, the ASCII horizontal space character is the only whitespace character that appears anywhere in a source file. Tab characters are used for indentation.
#### 2.3.2 Non-ASCII Characters
For the remaining non-ASCII characters, either the actual unicode character is used or the equivalent hex or unicode escape is used. This depends on what makes the code easier to understand. 

### 2.4 Naming Hierarchy
#### 2.4.1 Hierarchy
Module namespaces may never be named as a child of another module's namespace. The directory hierarchy reflects the namespace hierarchy. Thus, the deeper nested children are sub-directories of higher level parent directories.

### 2.5 Implementation
The actual implementation follows after all dependency information is declared. This may consist of any module-local declarations. 

## 3. Formatting
### 3.1 Block
Block-like construct refers to body of a class, function, method, or brace-delimited block of code. Object literal may be optionally treated as if it were a block-like construct
### 3.2 Braces
Braces are used for all control structures. Braces are required for all control structures, even if the body contains only a single statement.
### 3.3 Empty Blocks
An empty block may be closed immediately after it is opened, with no characters, space, or line break in between.
### 3.4 Block Indentation
Each time a new block or block-like construct is opened, the indent increases by two spaces. When the block ends, the indent returns to the previous indent level. The indent level applies to both code and comments throughout the block.
