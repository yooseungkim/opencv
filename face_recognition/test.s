dw_loop:
    %rdi, %rbx
    %rdi, %rcx
    $9, %rcx
    (,%rdi,4), %rdx
    movq
    movq
    idivq
    leaq 
.L2:
    5(%rbx,%rcx), %rcx $2, %rdx
    %rdx, %rdx
.L2
leaq subq testq
jg
rep; ret