; ModuleID = "module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"main"(i32* %"arr")
{
entry:
  %"ptr_0" = getelementptr i32, i32* %"arr", i32 0
  %"int_0" = load i32, i32* %"ptr_0"
  %"ptr_1" = getelementptr i32, i32* %"arr", i32 2
  store i32 %"int_0", i32* %"ptr_1"
  %"int_1" = load i32, i32* %"ptr_1"
  store i32 %"int_1", i32* %"ptr_1"
  %"int_2" = load i32, i32* %"ptr_1"
  %"bool_0" = icmp sgt i32 %"int_2", %"int_1"
  %"ptr_2" = getelementptr i32, i32* %"arr", i32 0
  br i1 %"bool_0", label %"true_0", label %"false_0"
true_0:
  %"int_3" = load i32, i32* %"ptr_2"
  ret void
false_0:
}
