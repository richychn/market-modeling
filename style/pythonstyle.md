# Python Coding Style Document

## 1. Background
Python is the main backend language used for Flask webapp and LSTM/ESN neural network models. This style guide is a list of dos and don'ts for Python programs.

## 2. Python Language Rules
### 2.1 Imports
Use import statements for packages and modules only, not for individual classes or functions. Note that there is an explicit exemption for imports from the typing module.
#### 2.1.1 Definition
Reusability mechanism for sharing code from one module to another.
#### 2.1.2 Pros
The namespace management convention is simple. The source of each identifier is indicated in a consistent way; x.Obj says that object Obj is defined in module x.
#### 2.1.3 Cons
Module names can still collide. Some module names are inconveniently long.
#### 2.1.4 Decision
Use import x for importing packages and modules.
Use from x import y where x is the package prefix and y is the module name with no prefix.
Use from x import y as z if two modules named y are to be imported or if y is an inconveniently long name.
Use import y as z only when z is a standard abbreviation (e.g., np for numpy).
For example the module keras.layers.LSTM may be imported as follows:
```
from keras.layers import LSTM
```
 
Do not use relative names in imports. Even if the module is in the same package, use the full package name. This helps prevent unintentionally importing a package twice.
 
### 2.2 Packages
Import each module using the full pathname location of the module.
#### 2.2.1 Pros
Avoids conflicts in module names or incorrect imports due to the module search path not being what the author expected. Makes it easier to find modules.
#### 2.2.2 Cons
Makes it harder to deploy code because you have to replicate the package hierarchy. Not really a problem with modern deployment mechanisms.
#### 2.2.3 Decision
All new code should import each module by its full package name.
Imports should be as follows:
Yes:
```python
# Reference absl.flags in code with the complete name (verbose).
from keras.layers import LSTM
model.add(LSTM(
    input_shape=(train_X.shape[1],train_X.shape[2]),
    return_sequences=True, units=100))
```
 
### 2.3 Global variables
Avoid global variables.
#### 2.3.1 Definition
Variables that are declared at the module level or as class attributes.
#### 2.3.2 Pros
Occasionally useful.
#### 2.3.3 Cons
Has the potential to change module behavior during the import, because assignments to global variables are done when the module is first imported.
#### 2.3.4 Decision
Avoid global variables.
While they are technically variables, module-level constants are permitted and encouraged. For example: MAX_LOOKBACK_COUNT = 3. Constants must be named using all caps with underscores. If needed, globals should be declared at the module level and made internal to the module by prepending an _ to the name. External access must be done through public module-level functions. 

### 2.4 Nested/Local/Inner Classes and Functions
Nested local functions or classes are fine when used to close over a local variable. Inner classes are fine.
#### 2.4.1 Definition
A class can be defined inside of a method, function, or class. A function can be defined inside a method or function. Nested functions have read-only access to variables defined in enclosing scopes.
#### 2.4.2 Pros
Allows definition of utility classes and functions that are only used inside of a very limited scope. 
#### 2.4.3 Cons
Instances of nested or local classes cannot be pickled. Nested functions and classes cannot be directly tested. Nesting can make your outer function longer and less readable.
#### 2.4.4 Decision
They are fine with some caveats. Avoid nested functions or classes except when closing over a local value. Do not nest a function just to hide it from users of a module. Instead, prefix its name with an _ at the module level so that it can still be accessed by tests.

### 2.5 Default Iterators and Operators
Use default iterators and operators for types that support them, like lists, dictionaries, and files.
#### 2.5.1 Definition
Container types, like dictionaries and lists, define default iterators and membership test operators ("in" and "not in").
#### 2.5.2 Pros
The default iterators and operators are simple and efficient. They express the operation directly, without extra method calls. A function that uses default operators is generic. It can be used with any type that supports the operation.
#### 2.5.3 Cons
You can't tell the type of objects by reading the method names (e.g. has_key() means a dictionary). This is also an advantage.
#### 2.5.4 Decision
Use default iterators and operators for types that support them, like lists, dictionaries, and files. The built-in types define iterator methods, too. Prefer these methods to methods that return lists, except that you should not mutate a container while iterating over it. Never use Python 2 specific iteration methods such as dict.iter*() unless necessary.

```
Yes:  for key in adict: ...
      if key not in adict: ...
      if obj in alist: ...
      for line in afile: ...
      for k, v in adict.items(): ...
      for k, v in six.iteritems(adict): ...
No:   for key in adict.keys(): ...
      if not adict.has_key(key): ...
      for line in afile.readlines(): ...
      for k, v in dict.iteritems(): ...
```

### 2.6 Default Argument Values
Okay in most cases.
#### 2.6.1 Definition
You can specify values for variables at the end of a function's parameter list, e.g., def foo(a, b=0):. If foo is called with only one argument, b is set to 0. If it is called with two arguments, b has the value of the second argument.
#### 2.6.2 Pros
Often you have a function that uses lots of default values, but-rarely-you want to override the defaults. Default argument values provide an easy way to do this, without having to define lots of functions for the rare exceptions. Also, Python does not support overloaded methods/functions and default arguments are an easy way of "faking" the overloading behavior.
#### 2.6.3 Cons
Default arguments are evaluated once at module load time. This may cause problems if the argument is a mutable object such as a list or a dictionary. If the function modifies the object (e.g., by appending an item to a list), the default value is modified.
#### 2.6.4 Decision
Okay to use with the following caveat:
Do not use mutable objects as default values in the function or method definition.

```
Yes: def foo(a, b=None):
         if b is None:
             b = []
Yes: def foo(a, b: Optional[Sequence] = None):
         if b is None:
             b = []
Yes: def foo(a, b: Sequence = ()):  # Empty tuple OK since tuples are immutable
No:  def foo(a, b=[]):
No:  def foo(a, b=time.time()):  # The time the module was loaded???
```
 
## 3 Python Style Rules
### 3.1 Semicolons
Do not terminate your lines with semicolons, and do not use semicolons to put two statements on the same line.

### 3.2 Line length
Maximum line length is 80 characters.

### 3.3 Indentation
Indent your code blocks with 4 spaces.

### 3.4 Whitespace
Follow standard typographic rules for the use of spaces around punctuation.
No whitespace inside parentheses, brackets or braces.
 
### 3.5 Naming
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME,global_var_name, instance_var_name, function_parameter_name, local_var_name.
Function names, variable names, and filenames should be descriptive; eschew abbreviation. In particular, do not use abbreviations that are ambiguous or unfamiliar to readers outside your project, and do not abbreviate by deleting letters within a word.
Always use a .py filename extension. Never use dashes.

## 4 Parting Words
BE CONSISTENT.
If you're editing code, take a few minutes to look at the code around you and determine its style. If they use spaces around all their arithmetic operators, you should too. If their comments have little boxes of hash marks around them, make your comments have little boxes of hash marks around them too.
 
 
 
 
 
 
 
 
 
