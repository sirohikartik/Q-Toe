const r11 = document.getElementById("11");
const r12 = document.getElementById("12");
const r13 = document.getElementById("13");

const r21 = document.getElementById("21");
const r22 = document.getElementById("22");
const r23 = document.getElementById("23");

const r31 = document.getElementById("31");
const r32 = document.getElementById("32");
const r33 = document.getElementById("33");
const record = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
];

const filled = [];
console.log(filled);

r11.addEventListener('click', () => {
    if (filled[0] != 1) {
        r11.textContent = "X";
        filled[0] = 1;
        record[0][0] = 1;  
        machine();
        checkwin();
    }
});

r12.addEventListener('click', () => {
    if (filled[1] != 1) {
        r12.textContent = "X";
        filled[1] = 1;
        record[0][1] = 1;  
       
        machine();
        checkwin();
    }
});

r13.addEventListener('click', () => {
    if (filled[2] != 1) {
        r13.textContent = "X";
        filled[2] = 1;
        record[0][2] = 1; 

        machine();
        checkwin();
    }
});

r21.addEventListener('click', () => {
    if (filled[3] != 1) {
        r21.textContent = "X";
        filled[3] = 1;
        record[1][0] = 1; 
       
            machine();
        checkwin();
  
        
    }
});

r22.addEventListener('click', () => {
    if (filled[4] != 1) {
        r22.textContent = "X";
        filled[4] = 1;
        record[1][1] = 1;  
    
            machine();
            checkwin();
    
     
    }
});

r23.addEventListener('click', () => {
    if (filled[5] != 1) {
        r23.textContent = "X";
        filled[5] = 1;
        record[1][2] = 1;  
     
            machine();
            checkwin();
    
    
    }
});

r31.addEventListener('click', () => {
    if (filled[6] != 1) {
        r31.textContent = "X";
        filled[6] = 1;
        record[2][0] = 1;  
     
            machine();
            checkwin();

      
    }
});

r32.addEventListener('click', () => {
    if (filled[7] != 1) {
        r32.textContent = "X";
        filled[7] = 1;
        record[2][1] = 1;  
          
    
            machine();
            checkwin();
        
       
    }
});

r33.addEventListener('click', () => {
    if (filled[8] != 1) {
        r33.textContent = "X";
        filled[8] = 1;
        record[2][2] = 1;
    
            machine();
            checkwin();
 
    }});

const boxes = [r11,r12,r13,r21,r22,r23,r31,r32,r33];

function machine() {
    let randomNumber;
    let count = 0;
    do {
        randomNumber = Math.floor(Math.random() * 9);
        console.log('ran');
        count+=1
        if(count>8) break;
    } while (filled[randomNumber] === 1);

    boxes[randomNumber].textContent = 'O';
    filled[randomNumber] = 1; 
}
let won = false;
function checkwin(){
    //linear row check:
    for(let i=0;i<3;i++)
    {
        let is1 = 1;
        for(let j=0;j<3;j++)
        {
            if(record[i][j]!=1) is1 = 0;
        }
        if(is1){
            won = true;
            console.log("won");
        }
    }

    //linear column check:
    for(let i=0;i<3;i++)
        {
            let is1 = 1;
            for(let j=0;j<3;j++)
            {
                if(record[j][i]!=1) is1 = 0;
            }
            if(is1){
                won = true;
                console.log("won");
            }
            
        }

    //Diagonal Check:
//First Diagonal:
if(record[0][0]==1 && record[1][1]==1 && record[2][2]==1)
{
    won = true;
    console.log("won");
}
if(record[0][2]==1 && record[1][1]==1 && record[2][0]==1)
    {
        won = true;
        console.log("won");
    }


}


