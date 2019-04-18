# HTML/CSS Coding Style Document

## 1. Background
This document defines formatting and style rules used for HTML and CSS. The purpose of this is to improve future collaboration, code quality, and efficiency. We used a basic framework and applied additional code to support future infrastructure. 

## 2. General
### 2.1 General Style Rules
Use appropriate protocols
#### 2.1.1 Protocol
Used HTTPS for embedded resources wherever possible. These include media files, style sheets, and scripts.
 
### 2.2 General Formatting Rules
Import each module using the full pathname location of the module.
#### 2.2.1 Indentation
Indent with 2 spaces (one tab) all throughout the front-end files. This includes all the HTML and CSS files on FLASK. 
Did not mix spaces and tabs.
#### 2.2.2 Capitalization
All code only uses lower case. This applies to HTML element names, attributes, classes, ID, CSS selectors, etc. 
Remove trailing white spaces as they can be unnecessary.
 
### 2.3 General Meta Rules
Includes encoding and comments.
#### 2.3.1 Encoding
The default character encoding in HTML5 is UTF-8. Moreover, HTML5 will not have a reference to the encoding in the doctype unless a web page uses a different character set.
#### 2.3.2 Comments
Explain code whenever needed.
Includes what the function does, what the purpose is, etc.
#### 2.3.3 Cons
Has the potential to change module behavior during the import, because assignments to global variables are done when the module is first imported.
 
## 3 HTML
### 3.1 Choice of Markup Language
Standard markup language was HTML.
HTML5 is prefered for all HTML documents.
Did not use XHTML as it was widely known that HTML5 has a much simpler doctype, offers more room for optimization, and better infrastructure support. 


### 3.2 Semantics
Used elements/tags/classes according to its purpose for accessibility, reuse, and code efficiency. 
Some examples include using heading elements for headings, paragraph tags for text, etc

### 3.3 Separation of Concern
Separated structure from presentation from behavior.
Strictly kept  structure (markup), presentation (styling), and behavior (scripting) apart and attempted to keep the interaction between them to a minimum. 
Kept everything related to structural purposes in the documents and templates.
Kept everything related to presentation in the style sheets.
Kept everything related to behavior in the script files.
Minimized contact and interaction between structure, style, and scripts by linking only a few style sheets from documents and templates.
Separating structure from presentation from behavior is important for maintenance reasons. One small modification may require us to make multiple changes should we not separate out the three areas. 

### 3.4 Formatting Rules
Use a new line for every new block, list, or table element and indent every child element in HTML files
If you run into issues around white space between list items, it is acceptable to put all elements in one line. 
 
### 3.5 Line Wrapping
Break long line (optional).
While there is no column limit recommendation for HTML, you may consider wrapping long lines if it significantly improves readability.
When line-wrapping, each continuation line should be indented with 2 spaces (one tab) from the original line.

### 3.6 HTML Quotation Marks
When quoting attributes values, use double quotation marks

## 4 CSS
### 4.1 CSS Validity
Use valid CSS code unless dealing with CSS validator bugs or requiring proprietary syntax.
Using valid CSS is a measurable baseline quality attribute that allows one to spot CSS code that may not have any effect and can be removed.

### 4.2 ID and Class Naming
Use meaningful and generic ID and class names.
Instead of cryptic names, always use ID and class names that reflect the purpose of the element in question or otherwise is generic.
Names that are specific and relate to the purpose of the element should be preferred as they are most understandable to foreign users. 
Generic names should only be used for elements that have no particular or no meaning different from their siblings.
The use of functional or generic names reduces the possibility of unnecessary changes to the template.
Use ID and class names that are short as possible but as long as necessary.
Attempt to convey what an ID or class is about while being as concise as possible.
Naming ID and class names in the ways listed above allows clear understandability and code efficiency.

### 4.3 Repetitive Naming/Selectors
Avoid any unnecessary ancestor selectors.

### 4.4 Hexadecimal Notation
Use 6 character hexadecimal notation to allow for a wide range of color values and refer to a specific color.

### 4.5 Indent Block Content
Indent all block content. 
This includes rules within rules as well as declarations.
This is in order to reflect hierarchy and improve understanding.

### 4.6 Declaration Stops
Use a semicolon after every declaration.
End every declaration with a semicolon for consistency and extensibility reasons.

### 4.7 Property Name Stops
Always use a single space between property and value for consistency reasons.
But no space between property and colon.

### 4.8 Comments
At the end of your implementation, go back to your code and group style sheet sections by using comments, if possible. 
Helps the individual review the written code and catch mistakes.

## 5 Parting Words
When you are editing code, look at the code around you and determine its style. If they use spaces around all their arithmetic operators, you should too. If their comments have little boxes of hash marks around them, make your comments have little boxes of hash marks around them too.
The point of having style guidelines is to have a common vocabulary of coding so that individuals can focus on what you are saying rather than how you are saying it. Consequently, we present the global style rules here so that people know the vocabulary. However, it is important to note that local style is important as well.
If you add code that looks dramatically different from the existing code around it, it will throw the readers out of their rhythm when they attempt to read it.

 
 
 
 
 
 
 
 
 

