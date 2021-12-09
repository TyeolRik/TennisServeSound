package main

import (
	"fmt"
	"os"

	"github.com/abema/go-mp4"
	"github.com/sunfish-shogi/bufseekio"
)

func main() {
	file, _ := os.Open("video_data/serve/good/good1 - flat (16).mp4")
	info, err := mp4.Probe(bufseekio.NewReadSeeker(file, 1024, 4))
	if err != nil {
		panic(err)
	}
	fmt.Println("track num:", info.Tracks[1])
}
